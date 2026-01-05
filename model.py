"""
FastConformer-Transducer Model

A streaming-capable, memory-efficient ASR model based on:
- FastConformer encoder (NVIDIA research)
- Stateless Transducer decoder
- Support for both offline and streaming inference

Reference: https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-06-07-fast-conformer/
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Subsampling8x(nn.Module):
    """
    8x subsampling module using depthwise separable convolutions.

    Reduces frame rate from 10ms to 80ms for efficient attention computation.
    Uses 256 hidden channels as per FastConformer paper.
    """

    def __init__(self, d_input: int = 80, d_model: int = 256, hidden_channels: int = 256):
        super().__init__()

        # First conv: 10ms -> 20ms (stride 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # Second conv: 20ms -> 40ms (stride 2)
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # Third conv: 40ms -> 80ms (stride 2)
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
        )

        # Calculate output dimension after 8x subsampling
        # After 3 convs with stride 2: freq_dim = ((d_input + 1) // 2 + 1) // 2 + 1) // 2
        freq_out = d_input
        for _ in range(3):
            freq_out = (freq_out + 1) // 2

        self.linear = nn.Linear(hidden_channels * freq_out, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, freq) mel spectrogram

        Returns:
            (batch, time//8, d_model)
        """
        # Add channel dimension: (batch, 1, time, freq)
        x = x.unsqueeze(1)

        # Apply convolutions
        x = self.conv1(x)  # (batch, hidden, time//2, freq//2)
        x = self.conv2(x)  # (batch, hidden, time//4, freq//4)
        x = self.conv3(x)  # (batch, hidden, time//8, freq//8)

        # Reshape: (batch, time//8, hidden * freq//8)
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * freq)

        # Project to d_model
        x = self.linear(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with support for both full and chunked attention.

    For streaming: uses chunked attention with left context cache.
    For offline: uses full bidirectional attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_pos_encoding: int = 5000,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Relative positional encoding
        self.pos_bias = nn.Parameter(torch.zeros(num_heads, max_pos_encoding * 2 - 1))
        nn.init.xavier_uniform_(self.pos_bias.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        mode: str = "offline",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            cache: Optional key-value cache for streaming
            mode: "offline" for full attention, "streaming" for chunked

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: Updated cache for streaming mode
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Handle cache for streaming mode
        new_cache = None
        if mode == "streaming" and cache is not None:
            # Concatenate cached keys and values
            cached_k, cached_v = cache.chunk(2, dim=-1)
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
            # Update cache
            new_cache = torch.cat([k, v], dim=-1)
        elif mode == "streaming":
            new_cache = torch.cat([k, v], dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)

        # For streaming, only return the new frames
        if mode == "streaming" and cache is not None:
            output = output[:, -seq_len:, :]

        return output, new_cache


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module with reduced kernel size (9 instead of 31).

    As per FastConformer: with 8x downsampling, smaller kernel preserves
    same receptive field while being more efficient.
    """

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expansion
        self.pointwise1 = nn.Conv1d(d_model, d_model * 2, 1)
        self.glu = nn.GLU(dim=1)

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()

        # Pointwise projection
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)

        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # (batch, seq_len, d_model)


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module."""

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class FastConformerBlock(nn.Module):
    """
    FastConformer block with streaming support.

    Structure: FFN -> MHSA -> Conv -> FFN (with 0.5 residual factor for FFNs)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ff_scale = 0.5

        self.ff1 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion_factor, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        mode: str = "offline",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            cache: Optional attention cache for streaming
            mode: "offline" or "streaming"

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: Updated cache
        """
        # First FFN with 0.5 residual
        x = x + self.ff_scale * self.ff1(x)

        # Self-attention
        attn_out, new_cache = self.attention(x, mask, cache, mode)
        x = x + attn_out

        # Convolution
        x = x + self.conv(x)

        # Second FFN with 0.5 residual
        x = x + self.ff_scale * self.ff2(x)

        # Final layer norm
        x = self.layer_norm(x)

        return x, new_cache


class FastConformerEncoder(nn.Module):
    """
    FastConformer Encoder with streaming support.

    Key optimizations from NVIDIA FastConformer:
    - 8x subsampling (vs 4x) for reduced attention cost
    - Depthwise separable convolutions
    - Reduced kernel size (9 vs 31)
    - Channel reduction in subsampling
    """

    def __init__(
        self,
        d_input: int = 80,
        d_model: int = 256,
        num_layers: int = 16,
        num_heads: int = 4,
        conv_kernel_size: int = 9,
        ff_expansion_factor: int = 4,
        dropout: float = 0.1,
        subsampling_channels: int = 256,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 8x subsampling
        self.subsampling = Subsampling8x(d_input, d_model, subsampling_channels)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # FastConformer blocks
        self.layers = nn.ModuleList([
            FastConformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        cache: Optional[List[torch.Tensor]] = None,
        mode: str = "offline",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, time, d_input) mel spectrogram
            lengths: (batch,) input lengths
            cache: List of cached states for each layer (streaming mode)
            mode: "offline" or "streaming"

        Returns:
            output: (batch, time//8, d_model)
            output_lengths: (batch,) output lengths
            new_cache: Updated cache list
        """
        # Subsampling
        x = self.subsampling(x)

        # Update lengths for 8x subsampling
        if lengths is not None:
            output_lengths = (lengths - 1) // 8 + 1
        else:
            output_lengths = torch.tensor([x.size(1)] * x.size(0), device=x.device)

        # Create attention mask for padding
        mask = None
        if lengths is not None:
            max_len = x.size(1)
            mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) >= output_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # Positional encoding
        x = self.pos_encoding(x)

        # Initialize cache if streaming
        new_cache = []
        if cache is None:
            cache = [None] * self.num_layers

        # Apply FastConformer blocks
        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, cache[i], mode)
            new_cache.append(layer_cache)

        return x, output_lengths, new_cache if mode == "streaming" else None


class StatelessPredictor(nn.Module):
    """
    Stateless prediction network for Transducer.

    Uses embedding + causal Conv1D instead of LSTM.
    Much more efficient for streaming as there's no hidden state to maintain.
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
            self.convs.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size - 1),
                nn.SiLU(),
                nn.Dropout(dropout),
            ))

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, d_model) predictor output
        """
        # Embedding
        x = self.embedding(y)
        x = self.embed_dropout(x)

        # Causal convolutions
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        for conv in self.convs:
            residual = x
            x = conv(x)
            # Ensure causality by removing future frames
            x = x[:, :, :residual.size(2)]
            x = x + residual

        x = x.transpose(1, 2)  # (batch, seq_len, d_model)
        x = self.layer_norm(x)

        return x


class JointNetwork(nn.Module):
    """
    Joint network that combines encoder and predictor outputs.

    Computes: joint(enc, pred) = Linear(tanh(Linear(enc) + Linear(pred)))
    """

    def __init__(
        self,
        d_encoder: int,
        d_predictor: int,
        d_joint: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = nn.Linear(d_encoder, d_joint)
        self.predictor_proj = nn.Linear(d_predictor, d_joint)
        self.output_proj = nn.Linear(d_joint, vocab_size)

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
            (batch, T, U, vocab_size) joint output logits
        """
        # Project encoder: (batch, T, 1, d_joint)
        enc = self.encoder_proj(encoder_out).unsqueeze(2)

        # Project predictor: (batch, 1, U, d_joint)
        pred = self.predictor_proj(predictor_out).unsqueeze(1)

        # Combine with broadcasting: (batch, T, U, d_joint)
        joint = torch.tanh(enc + pred)

        # Output projection: (batch, T, U, vocab_size)
        output = self.output_proj(joint)

        return output


class FastConformerTransducer(nn.Module):
    """
    FastConformer-Transducer model for streaming ASR.

    Combines:
    - FastConformer encoder (efficient, streaming-capable)
    - Stateless predictor (no hidden state)
    - Joint network for Transducer output

    Supports both offline and streaming inference modes.
    """

    def __init__(
        self,
        d_input: int = 80,
        vocab_size: int = 29,
        d_encoder: int = 256,
        d_predictor: int = 256,
        d_joint: int = 320,
        encoder_layers: int = 16,
        encoder_heads: int = 4,
        encoder_conv_kernel: int = 9,
        encoder_ff_expansion: int = 4,
        predictor_layers: int = 2,
        predictor_kernel: int = 2,
        dropout: float = 0.1,
        subsampling_channels: int = 256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.blank_id = 0  # Blank token for RNN-T

        self.encoder = FastConformerEncoder(
            d_input=d_input,
            d_model=d_encoder,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            conv_kernel_size=encoder_conv_kernel,
            ff_expansion_factor=encoder_ff_expansion,
            dropout=dropout,
            subsampling_channels=subsampling_channels,
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
        mode: str = "offline",
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            audio: (batch, time, d_input) mel spectrogram
            audio_lengths: (batch,) input lengths
            targets: (batch, max_target_len) target token indices
            target_lengths: (batch,) target lengths
            mode: "offline" or "streaming"

        Returns:
            logits: (batch, T, U, vocab_size) joint output for RNN-T loss
        """
        # Encode audio
        encoder_out, encoder_lengths, _ = self.encoder(audio, audio_lengths, mode=mode)

        # Prepend blank to targets for predictor input
        batch_size = targets.size(0)
        blank = torch.zeros(batch_size, 1, dtype=targets.dtype, device=targets.device)
        predictor_input = torch.cat([blank, targets], dim=1)

        # Predict from previous tokens
        predictor_out = self.predictor(predictor_input)

        # Joint network
        logits = self.joint(encoder_out, predictor_out)

        return logits, encoder_lengths

    def encode(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        cache: Optional[List[torch.Tensor]] = None,
        mode: str = "offline",
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Encode audio features.

        Args:
            audio: (batch, time, d_input) mel spectrogram
            audio_lengths: (batch,) input lengths
            cache: Encoder cache for streaming
            mode: "offline" or "streaming"

        Returns:
            encoder_out: (batch, T, d_encoder)
            encoder_lengths: (batch,)
            new_cache: Updated cache
        """
        return self.encoder(audio, audio_lengths, cache, mode)

    def predict(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Run predictor on target sequence.

        Args:
            targets: (batch, seq_len) token indices

        Returns:
            (batch, seq_len, d_predictor)
        """
        return self.predictor(targets)

    def joint_step(
        self,
        encoder_out: torch.Tensor,
        predictor_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single step joint network computation.

        Args:
            encoder_out: (batch, 1, d_encoder) single frame
            predictor_out: (batch, 1, d_predictor) single prediction

        Returns:
            (batch, vocab_size) logits
        """
        # (batch, 1, 1, vocab_size)
        logits = self.joint(encoder_out, predictor_out)
        # (batch, vocab_size)
        return logits.squeeze(1).squeeze(1)


def create_model(config: dict) -> FastConformerTransducer:
    """
    Create FastConformerTransducer model from config.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        FastConformerTransducer model
    """
    model_config = config['model']

    return FastConformerTransducer(
        d_input=model_config.get('d_input', 80),
        vocab_size=model_config.get('vocab_size', 29),
        d_encoder=model_config.get('d_encoder', 256),
        d_predictor=model_config.get('d_predictor', 256),
        d_joint=model_config.get('d_joint', 320),
        encoder_layers=model_config.get('encoder_layers', 16),
        encoder_heads=model_config.get('encoder_heads', 4),
        encoder_conv_kernel=model_config.get('encoder_conv_kernel', 9),
        encoder_ff_expansion=model_config.get('encoder_ff_expansion', 4),
        predictor_layers=model_config.get('predictor_layers', 2),
        predictor_kernel=model_config.get('predictor_kernel', 2),
        dropout=model_config.get('dropout', 0.1),
        subsampling_channels=model_config.get('subsampling_channels', 256),
    )


def model_size(model: nn.Module, name: str = "Model") -> dict:
    """
    Calculate and print model size.

    Returns dict with parameter count and size in MB.
    """
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
