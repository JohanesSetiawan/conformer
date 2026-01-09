"""
FastConformer-Transducer Model (NVIDIA-style Implementation)

A streaming-capable, low-latency ASR model optimized for 80ms latency.

Key Features:
- Relative Positional Encoding (Transformer-XL style)
- Causal Convolution with caching for true streaming
- Local Attention with sliding window for constant memory
- Stochastic Depth regularization
- Cache-aware streaming configuration

Reference:
- https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-06-07-fast-conformer/
- https://arxiv.org/abs/2511.23404 (LFM2 Technical Report)
"""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StreamingConfig:
    """Configuration for cache-aware streaming inference."""
    chunk_size: int = 8          # Frames per chunk (8 frames = 80ms at 8x subsampling)
    left_context: int = 32       # Left context frames for attention
    right_context: int = 0       # Right context (0 for true streaming)
    max_cache_len: int = 512     # Maximum cache length (prevents memory growth)

    @property
    def total_context(self) -> int:
        return self.left_context + self.right_context


@dataclass
class StochasticDepthConfig:
    """Configuration for stochastic depth regularization."""
    enabled: bool = True
    drop_prob: float = 0.1       # Maximum drop probability
    mode: str = "linear"         # "linear" or "uniform"
    start_layer: int = 1         # Layer to start applying (1-indexed)


# =============================================================================
# Utility Functions
# =============================================================================

def compute_stochastic_depth_drop_probs(
    num_layers: int,
    drop_prob: float = 0.1,
    mode: str = "linear",
    start_layer: int = 1,
) -> List[float]:
    """
    Compute per-layer drop probabilities for stochastic depth.

    Args:
        num_layers: Total number of layers
        drop_prob: Maximum drop probability
        mode: "linear" (gradual increase) or "uniform" (same for all)
        start_layer: Layer to start applying (1-indexed)

    Returns:
        List of drop probabilities per layer
    """
    if not (0 <= drop_prob < 1.0):
        raise ValueError("drop_prob must be in [0, 1)")

    # First start_layer-1 layers have 0 drop prob
    probs = [0.0] * (start_layer - 1)

    remaining = num_layers - start_layer + 1
    if remaining > 0:
        if mode == "linear":
            # Linearly increase from 0 to drop_prob
            probs += [i / remaining * drop_prob for i in range(1, remaining + 1)]
        elif mode == "uniform":
            probs += [drop_prob] * remaining
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return probs


def avoid_float16_autocast_context():
    """Avoid float16 autocast issues by switching to bfloat16 or float32."""
    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        if torch.cuda.is_bf16_supported():
            return torch.amp.autocast('cuda', dtype=torch.bfloat16)
        return torch.amp.autocast('cuda', dtype=torch.float32)
    return nullcontext()


# =============================================================================
# Relative Positional Encoding (Transformer-XL style)
# =============================================================================

class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding as used in Transformer-XL.

    Reference: https://arxiv.org/abs/1901.02860

    Creates bidirectional position encodings from -(max_len-1) to (max_len-1).
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        scale: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model) if scale else 1.0

        # Create initial positional encoding
        self._create_pe(max_len)

    def _create_pe(self, max_len: int):
        """Create sinusoidal positional encodings."""
        # Positions from -(max_len-1) to (max_len-1)
        positions = torch.arange(-(max_len - 1), max_len, dtype=torch.float32)

        # Div term for sinusoidal encoding
        dim = torch.arange(0, self.d_model, 2, dtype=torch.float32)
        div_term = torch.exp(dim * (-math.log(10000.0) / self.d_model))

        # Create encoding matrix
        pe = torch.zeros(2 * max_len - 1, self.d_model)
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)

        # Update or create buffer
        if hasattr(self, 'pe') and self.pe is not None:
            del self.pe
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def extend_pe(self, length: int):
        """Extend positional encoding if needed."""
        if not hasattr(self, 'pe') or self.pe is None or length > self.max_len:
            self._create_pe(max(length, self.max_len * 2))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x_scaled: Scaled input
            pos_emb: (seq_len * 2 - 1, d_model) relative position embeddings
        """
        seq_len = x.size(1)
        self.extend_pe(seq_len)

        # Scale input
        x = x * self.scale
        x = self.dropout(x)

        # Get relative position embeddings centered at current position
        # For seq_len=T, we need positions from -(T-1) to (T-1)
        center = self.max_len - 1
        start = center - seq_len + 1
        end = center + seq_len
        pos_emb = self.pe[start:end]

        return x, pos_emb


# =============================================================================
# Causal Convolution with Cache
# =============================================================================

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with cache support for streaming.

    Outputs only depend on current and past inputs, never future.
    Cache allows incremental computation without reprocessing history.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        # Cache size = receptive field - 1
        self.cache_size = (kernel_size - 1) * dilation

        # No padding - we'll handle it with cache
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=0, dilation=dilation,
            groups=groups, bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, channels, time)
            cache: (batch, channels, cache_size) from previous chunk

        Returns:
            output: (batch, out_channels, time)
            new_cache: (batch, channels, cache_size) for next chunk
        """
        if cache is not None:
            # Prepend cache for causal context
            x = torch.cat([cache, x], dim=-1)
        else:
            # First chunk: pad with zeros
            x = F.pad(x, (self.cache_size, 0))

        # Save new cache (last cache_size frames of input)
        new_cache = x[:, :, -self.cache_size:].clone() if self.cache_size > 0 else None

        # Apply convolution
        output = self.conv(x)

        return output, new_cache

    def get_initial_cache(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create zero-initialized cache."""
        return torch.zeros(
            batch_size, self.conv.in_channels, self.cache_size,
            device=device
        )


# =============================================================================
# Multi-Head Attention with Relative Position
# =============================================================================

class RelPositionMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.

    Implements the attention mechanism from Transformer-XL with:
    - Content-based attention (Q @ K)
    - Position-based attention (Q @ pos_emb)
    - Learnable position biases (u and v)

    Supports both full attention and local (sliding window) attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_pos_encoding: int = 5000,
        use_local_attn: bool = False,
        local_context: int = 256,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.use_local_attn = use_local_attn
        self.local_context = local_context

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Position projection
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable biases for position attention (Transformer-XL style)
        self.pos_bias_u = nn.Parameter(torch.zeros(num_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.zeros(num_heads, self.d_head))
        nn.init.xavier_uniform_(self.pos_bias_u.unsqueeze(0))
        nn.init.xavier_uniform_(self.pos_bias_v.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Relative position shift for efficient computation.

        Transforms the position attention matrix to align positions correctly.
        """
        b, h, qlen, klen = x.size()

        # Pad and reshape to shift
        x = F.pad(x, (1, 0))  # (b, h, qlen, klen+1)
        x = x.view(b, h, klen + 1, qlen)
        x = x[:, :, 1:, :].view(b, h, qlen, klen)

        return x

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            pos_emb: (2*seq_len-1, d_model) relative position embeddings
            mask: Optional attention mask
            cache: Optional KV cache for streaming

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: Updated KV cache
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Handle KV cache for streaming
        new_cache = None
        if cache is not None:
            cached_k, cached_v = cache.chunk(2, dim=-1)
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

            # Limit cache size for constant memory
            if k.size(1) > self.local_context:
                k = k[:, -self.local_context:]
                v = v[:, -self.local_context:]

            new_cache = torch.cat([k, v], dim=-1)
        else:
            new_cache = torch.cat([k, v], dim=-1)

        kv_len = k.size(1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head)
        k = k.view(batch_size, kv_len, self.num_heads, self.d_head)
        v = v.view(batch_size, kv_len, self.num_heads, self.d_head)

        # Position projection
        p = self.pos_proj(pos_emb)  # (2*seq_len-1, d_model)
        p = p.view(-1, self.num_heads, self.d_head)  # (2*seq_len-1, heads, d_head)

        # Content attention: (Q + u) @ K
        q_with_u = q + self.pos_bias_u  # (batch, seq, heads, d_head)
        q_with_u = q_with_u.transpose(1, 2)  # (batch, heads, seq, d_head)
        k_t = k.permute(0, 2, 3, 1)  # (batch, heads, d_head, kv_len)
        content_score = torch.matmul(q_with_u, k_t)  # (batch, heads, seq, kv_len)

        # Position attention: (Q + v) @ P
        q_with_v = q + self.pos_bias_v
        q_with_v = q_with_v.transpose(1, 2)  # (batch, heads, seq, d_head)
        p_t = p.permute(1, 2, 0)  # (heads, d_head, 2*seq-1)
        pos_score = torch.matmul(q_with_v, p_t)  # (batch, heads, seq, 2*seq-1)

        # Apply relative shift to align positions
        pos_score = self._rel_shift(pos_score)

        # Handle size mismatch for cached attention
        if pos_score.size(-1) > kv_len:
            pos_score = pos_score[:, :, :, :kv_len]
        elif pos_score.size(-1) < kv_len:
            # Pad position scores if cache is longer
            pad_len = kv_len - pos_score.size(-1)
            pos_score = F.pad(pos_score, (pad_len, 0))

        # Combined attention scores
        attn = (content_score + pos_score) * self.scale

        # Apply local attention mask if enabled
        if self.use_local_attn and kv_len > self.local_context:
            local_mask = torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, kv_len - seq_len + i - self.local_context + 1)
                end = kv_len - seq_len + i + 1
                local_mask[i, start:end] = False
            attn = attn.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply padding mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        v = v.transpose(1, 2)  # (batch, heads, kv_len, d_head)
        output = torch.matmul(attn, v)  # (batch, heads, seq, d_head)
        output = output.transpose(1, 2).contiguous()  # (batch, seq, heads, d_head)
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output, new_cache


# =============================================================================
# Conformer Convolution Module (Causal)
# =============================================================================

class ConformerConvolution(nn.Module):
    """
    Conformer convolution module with causal support for streaming.

    Structure:
    - LayerNorm
    - Pointwise Conv (expansion)
    - GLU activation
    - Depthwise Causal Conv
    - BatchNorm / LayerNorm (configurable)
    - SiLU activation
    - Pointwise Conv (projection)
    - Dropout
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 9,
        dropout: float = 0.1,
        norm_type: str = "batch",  # "batch", "layer", "instance", "group"
        use_causal: bool = True,
    ):
        super().__init__()
        self.use_causal = use_causal
        self.kernel_size = kernel_size

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expansion (2x for GLU)
        self.pointwise1 = nn.Conv1d(d_model, d_model * 2, 1)
        self.glu = nn.GLU(dim=1)

        # Depthwise convolution (causal or non-causal)
        if use_causal:
            self.depthwise = CausalConv1d(
                d_model, d_model, kernel_size, groups=d_model
            )
        else:
            padding = (kernel_size - 1) // 2
            self.depthwise = nn.Conv1d(
                d_model, d_model, kernel_size,
                padding=padding, groups=d_model
            )

        # Normalization
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(d_model)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(d_model)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(8, d_model)
        else:
            self.norm = nn.Identity()

        self.norm_type = norm_type
        self.activation = nn.SiLU()

        # Pointwise projection
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            cache: Conv cache for streaming

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: Updated conv cache
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)

        # Pointwise expansion + GLU
        x = self.pointwise1(x)
        x = self.glu(x)

        # Depthwise convolution
        new_cache = None
        if self.use_causal:
            x, new_cache = self.depthwise(x, cache)
        else:
            x = self.depthwise(x)

        # Normalization
        if self.norm_type == "layer":
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)

        x = self.activation(x)

        # Pointwise projection
        x = self.pointwise2(x)
        x = self.dropout(x)

        return x.transpose(1, 2), new_cache

    def get_initial_cache(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """Get initial conv cache."""
        if self.use_causal:
            return self.depthwise.get_initial_cache(batch_size, device)
        return None


# =============================================================================
# Feed-Forward Module
# =============================================================================

class FeedForwardModule(nn.Module):
    """Conformer feed-forward module with SiLU activation."""

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        d_ff = d_model * expansion_factor

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


# =============================================================================
# FastConformer Block
# =============================================================================

class FastConformerBlock(nn.Module):
    """
    FastConformer block with streaming support and stochastic depth.

    Structure: FFN(0.5) -> MHSA -> Conv -> FFN(0.5) -> LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        use_local_attn: bool = False,
        local_context: int = 256,
        stochastic_depth_prob: float = 0.0,
        conv_norm_type: str = "batch",
        use_causal_conv: bool = True,
    ):
        super().__init__()

        self.ff_scale = 0.5
        self.drop_prob = stochastic_depth_prob

        self.ff1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)

        self.attention = RelPositionMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_local_attn=use_local_attn,
            local_context=local_context,
        )

        self.conv = ConformerConvolution(
            d_model=d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            norm_type=conv_norm_type,
            use_causal=use_causal_conv,
        )

        self.ff2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        attn_cache: Optional[torch.Tensor] = None,
        conv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            pos_emb: Relative position embeddings
            mask: Attention mask
            attn_cache: Attention KV cache
            conv_cache: Convolution cache

        Returns:
            output: (batch, seq_len, d_model)
            new_attn_cache: Updated attention cache
            new_conv_cache: Updated convolution cache
        """
        # Stochastic depth: skip entire block during training
        if self.training and self.drop_prob > 0 and random.random() < self.drop_prob:
            return x, attn_cache, conv_cache

        # First FFN with 0.5 residual
        x = x + self.ff_scale * self.ff1(x)

        # Self-attention with relative position
        attn_out, new_attn_cache = self.attention(x, pos_emb, mask, attn_cache)
        x = x + attn_out

        # Convolution
        conv_out, new_conv_cache = self.conv(x, conv_cache)
        x = x + conv_out

        # Second FFN with 0.5 residual
        x = x + self.ff_scale * self.ff2(x)

        # Final layer norm
        x = self.layer_norm(x)

        return x, new_attn_cache, new_conv_cache


# =============================================================================
# Subsampling Module
# =============================================================================

class Subsampling8x(nn.Module):
    """
    8x subsampling using depthwise separable convolutions.

    Reduces 10ms frames to 80ms frames (8x downsampling).
    Supports both streaming and offline modes.
    """

    def __init__(
        self,
        d_input: int = 80,
        d_model: int = 256,
        hidden_channels: int = 256,
    ):
        super().__init__()

        # Conv stack: 3 layers with stride 2 each = 8x downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # Calculate output frequency dimension
        freq_out = d_input
        for _ in range(3):
            freq_out = (freq_out + 1) // 2

        self.linear = nn.Linear(hidden_channels * freq_out, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, freq) mel spectrogram

        Returns:
            (batch, time//8, d_model)
        """
        # Add channel dim: (batch, 1, time, freq)
        x = x.unsqueeze(1)

        # Apply conv stack
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Reshape: (batch, time//8, hidden * freq//8)
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * freq)

        # Project to d_model
        x = self.linear(x)

        return x

    def get_output_length(self, input_length: int) -> int:
        """Calculate output length for given input length."""
        length = input_length
        for _ in range(3):
            length = (length + 1) // 2
        return length


# =============================================================================
# FastConformer Encoder
# =============================================================================

class FastConformerEncoder(nn.Module):
    """
    FastConformer Encoder with full streaming support.

    Features:
    - 8x subsampling for efficient attention
    - Relative positional encoding
    - Local attention option for constant memory
    - Stochastic depth regularization
    - Per-layer caching for streaming
    """

    def __init__(
        self,
        d_input: int = 80,
        d_model: int = 256,
        num_layers: int = 17,
        num_heads: int = 4,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        subsampling_channels: int = 256,
        use_local_attn: bool = False,
        local_context: int = 256,
        stochastic_depth: Optional[StochasticDepthConfig] = None,
        conv_norm_type: str = "batch",
        use_causal_conv: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_causal_conv = use_causal_conv

        # Subsampling
        self.subsampling = Subsampling8x(d_input, d_model, subsampling_channels)

        # Relative positional encoding
        self.pos_encoding = RelativePositionalEncoding(d_model, dropout=dropout)

        # Compute stochastic depth probabilities
        if stochastic_depth and stochastic_depth.enabled:
            drop_probs = compute_stochastic_depth_drop_probs(
                num_layers,
                stochastic_depth.drop_prob,
                stochastic_depth.mode,
                stochastic_depth.start_layer,
            )
        else:
            drop_probs = [0.0] * num_layers

        # FastConformer blocks
        self.layers = nn.ModuleList([
            FastConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                dropout=dropout,
                use_local_attn=use_local_attn,
                local_context=local_context,
                stochastic_depth_prob=drop_probs[i],
                conv_norm_type=conv_norm_type,
                use_causal_conv=use_causal_conv,
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            x: (batch, time, d_input) mel spectrogram
            lengths: (batch,) input lengths (pre-subsampling)
            cache: List of (attn_cache, conv_cache) per layer
            streaming: Enable streaming mode

        Returns:
            output: (batch, time//8, d_model)
            output_lengths: (batch,) output lengths
            new_cache: Updated cache list
        """
        # Subsampling
        x = self.subsampling(x)

        # Compute output lengths
        if lengths is not None:
            output_lengths = torch.tensor(
                [self.subsampling.get_output_length(l.item()) for l in lengths],
                device=x.device
            )
        else:
            output_lengths = torch.tensor([x.size(1)] * x.size(0), device=x.device)

        # Create padding mask
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1)
            mask = mask >= output_lengths.unsqueeze(1)

        # Apply relative positional encoding
        x, pos_emb = self.pos_encoding(x)

        # Initialize cache
        new_cache = []
        if cache is None:
            cache = [(None, None)] * self.num_layers

        # Apply FastConformer blocks
        for i, layer in enumerate(self.layers):
            attn_cache, conv_cache = cache[i]
            x, new_attn_cache, new_conv_cache = layer(
                x, pos_emb, mask, attn_cache, conv_cache
            )
            new_cache.append((new_attn_cache, new_conv_cache))

        return x, output_lengths, new_cache if streaming else None

    def get_initial_cache(self, batch_size: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create initial cache for streaming."""
        cache = []
        for layer in self.layers:
            conv_cache = layer.conv.get_initial_cache(batch_size, device)
            cache.append((None, conv_cache))
        return cache


# =============================================================================
# Stateless Predictor
# =============================================================================

class StatelessPredictor(nn.Module):
    """
    Stateless prediction network using Conv1D instead of LSTM.

    No hidden state to maintain, making it efficient for streaming.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 2,
        kernel_size: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Causal Conv1D layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                CausalConv1d(d_model, d_model, kernel_size)
            )

        self.activation = nn.SiLU()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, d_model)
        """
        x = self.embedding(y)
        x = self.embed_dropout(x)

        # Apply causal convolutions
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        for conv in self.convs:
            residual = x
            x, _ = conv(x, None)
            x = self.activation(x)
            x = x + residual

        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.layer_norm(x)

        return x


# =============================================================================
# Joint Network
# =============================================================================

class JointNetwork(nn.Module):
    """Joint network combining encoder and predictor outputs."""

    def __init__(
        self,
        d_encoder: int,
        d_predictor: int,
        d_joint: int,
        vocab_size: int,
        logit_clamp: float = 15.0,  # Clamp logits for numerical stability
    ):
        super().__init__()

        self.encoder_proj = nn.Linear(d_encoder, d_joint)
        self.predictor_proj = nn.Linear(d_predictor, d_joint)
        self.output_proj = nn.Linear(d_joint, vocab_size)
        self.logit_clamp = logit_clamp

    def forward(
        self,
        encoder_out: torch.Tensor,
        predictor_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: (batch, T, d_encoder)
            predictor_out: (batch, U, d_predictor)

        Returns:
            (batch, T, U, vocab_size)
        """
        enc = self.encoder_proj(encoder_out).unsqueeze(2)
        pred = self.predictor_proj(predictor_out).unsqueeze(1)
        joint = torch.tanh(enc + pred)
        logits = self.output_proj(joint)

        # Clamp logits to prevent overflow in softmax/loss computation
        if self.logit_clamp > 0:
            logits = logits.clamp(-self.logit_clamp, self.logit_clamp)

        return logits


# =============================================================================
# FastConformer-Transducer Model
# =============================================================================

class FastConformerTransducer(nn.Module):
    """
    FastConformer-Transducer model optimized for low-latency streaming ASR.

    Target latency: ~80ms (1 encoder frame at 8x subsampling)

    Features:
    - FastConformer encoder with relative positional encoding
    - Causal convolutions for true streaming
    - Local attention for constant memory usage
    - Stochastic depth for better generalization
    - Stateless predictor (no hidden state)
    """

    def __init__(
        self,
        d_input: int = 80,
        vocab_size: int = 29,
        d_encoder: int = 256,
        d_predictor: int = 256,
        d_joint: int = 320,
        encoder_layers: int = 17,
        encoder_heads: int = 4,
        encoder_conv_kernel: int = 9,
        encoder_ff_expansion: int = 4,
        predictor_layers: int = 2,
        predictor_kernel: int = 2,
        dropout: float = 0.1,
        subsampling_channels: int = 256,
        use_local_attn: bool = True,
        local_context: int = 256,
        stochastic_depth_prob: float = 0.1,
        conv_norm_type: str = "batch",
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.blank_id = 0
        self.d_encoder = d_encoder

        # Stochastic depth config
        sd_config = StochasticDepthConfig(
            enabled=stochastic_depth_prob > 0,
            drop_prob=stochastic_depth_prob,
            mode="linear",
            start_layer=1,
        )

        self.encoder = FastConformerEncoder(
            d_input=d_input,
            d_model=d_encoder,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            conv_kernel_size=encoder_conv_kernel,
            ff_expansion_factor=encoder_ff_expansion,
            dropout=dropout,
            subsampling_channels=subsampling_channels,
            use_local_attn=use_local_attn,
            local_context=local_context,
            stochastic_depth=sd_config,
            conv_norm_type=conv_norm_type,
            use_causal_conv=True,
        )

        self.predictor = StatelessPredictor(
            vocab_size=vocab_size,
            d_model=d_predictor,
            num_layers=predictor_layers,
            kernel_size=predictor_kernel,
            dropout=dropout,
        )

        self.joint = JointNetwork(
            d_encoder=d_encoder,
            d_predictor=d_predictor,
            d_joint=d_joint,
            vocab_size=vocab_size,
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            audio: (batch, time, d_input)
            audio_lengths: (batch,)
            targets: (batch, max_target_len)
            target_lengths: (batch,)
            debug: Enable debug output for tensor statistics

        Returns:
            logits: (batch, T, U+1, vocab_size)
            encoder_lengths: (batch,)
        """
        if debug:
            self._debug_tensor(audio, "input.audio")

        # Encode
        encoder_out, encoder_lengths, _ = self.encoder(audio, audio_lengths)

        if debug:
            self._debug_tensor(encoder_out, "encoder.output")

        # Prepend blank for predictor
        batch_size = targets.size(0)
        blank = torch.zeros(batch_size, 1, dtype=targets.dtype, device=targets.device)
        predictor_input = torch.cat([blank, targets], dim=1)

        # Predict
        predictor_out = self.predictor(predictor_input)

        if debug:
            self._debug_tensor(predictor_out, "predictor.output")

        # Joint
        logits = self.joint(encoder_out, predictor_out)

        if debug:
            self._debug_tensor(logits, "joint.logits")

        return logits, encoder_lengths

    def _debug_tensor(self, tensor: torch.Tensor, name: str):
        """Print tensor debug statistics."""
        with torch.no_grad():
            t = tensor.float()
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()

            finite_mask = torch.isfinite(t)
            if finite_mask.any():
                finite_vals = t[finite_mask]
                stats = {
                    "min": finite_vals.min().item(),
                    "max": finite_vals.max().item(),
                    "mean": finite_vals.mean().item(),
                    "std": finite_vals.std().item() if finite_vals.numel() > 1 else 0,
                    "abs_max": finite_vals.abs().max().item(),
                }
            else:
                stats = {"min": float('nan'), "max": float('nan'), "mean": float('nan'), "std": float('nan'), "abs_max": float('nan')}

            status = "OK"
            if has_nan:
                status = f"NaN({torch.isnan(t).sum().item()})"
            elif has_inf:
                status = f"Inf({torch.isinf(t).sum().item()})"

            print(
                f"[DEBUG] {name:30s} | shape={list(tensor.shape)} | "
                f"min={stats['min']:+.4e} max={stats['max']:+.4e} | "
                f"mean={stats['mean']:+.4e} std={stats['std']:.4e} | "
                f"abs_max={stats['abs_max']:.4e} | {status}"
            )

    def encode(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Encode audio features."""
        return self.encoder(audio, audio_lengths, cache, streaming)

    def predict(self, targets: torch.Tensor) -> torch.Tensor:
        """Run predictor on target sequence."""
        return self.predictor(targets)

    def joint_step(
        self,
        encoder_out: torch.Tensor,
        predictor_out: torch.Tensor,
    ) -> torch.Tensor:
        """Single step joint computation."""
        logits = self.joint(encoder_out, predictor_out)
        return logits.squeeze(1).squeeze(1)

    def get_initial_cache(
        self,
        batch_size: int,
        device: torch.device,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get initial encoder cache for streaming."""
        return self.encoder.get_initial_cache(batch_size, device)


# =============================================================================
# Model Creation and Utilities
# =============================================================================

def create_model(config: dict) -> FastConformerTransducer:
    """Create model from config dictionary."""
    model_config = config.get('model', {})

    return FastConformerTransducer(
        d_input=model_config.get('d_input', 80),
        vocab_size=model_config.get('vocab_size', 29),
        d_encoder=model_config.get('d_encoder', 256),
        d_predictor=model_config.get('d_predictor', 256),
        d_joint=model_config.get('d_joint', 320),
        encoder_layers=model_config.get('encoder_layers', 17),
        encoder_heads=model_config.get('encoder_heads', 4),
        encoder_conv_kernel=model_config.get('encoder_conv_kernel', 9),
        encoder_ff_expansion=model_config.get('encoder_ff_expansion', 4),
        predictor_layers=model_config.get('predictor_layers', 2),
        predictor_kernel=model_config.get('predictor_kernel', 2),
        dropout=model_config.get('dropout', 0.1),
        subsampling_channels=model_config.get('subsampling_channels', 256),
        use_local_attn=model_config.get('use_local_attn', True),
        local_context=model_config.get('local_context', 256),
        stochastic_depth_prob=model_config.get('stochastic_depth_prob', 0.1),
        conv_norm_type=model_config.get('conv_norm_type', 'batch'),
    )


def model_size(model: nn.Module, name: str = "Model") -> dict:
    """Calculate and print model size."""
    param_size = 0
    num_params = 0

    for param in model.parameters():
        num_params += param.numel()
        param_size += param.numel() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)

    print(f"{name}: {num_params / 1e6:.2f}M parameters, {size_mb:.2f} MB")

    return {
        'num_params': num_params,
        'size_mb': size_mb,
    }
