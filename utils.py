"""
FastConformer-Transducer Utilities

Utility functions and classes for training and inference.
"""

import os
import json
import random
import gc
import torch
import torch.nn as nn
import torch.distributed as dist


class TextTransform:
    """
    Map characters to integers and vice versa.

    Character set: A-Z, apostrophe, space (28 characters + blank = 29 vocab)
    Blank token (index 0) is reserved for RNN-T.
    """

    def __init__(self):
        # Reserve index 0 for blank token
        self.blank_id = 0

        # Character to index mapping (indices 1-28)
        self.char_map = {}
        for i, char in enumerate(range(65, 91)):  # A-Z
            self.char_map[chr(char)] = i + 1  # Start from 1, not 0
        self.char_map["'"] = 27
        self.char_map[' '] = 28

        # Index to character mapping
        self.index_map = {v: k for k, v in self.char_map.items()}
        self.index_map[0] = ''  # Blank

        self.vocab_size = 29  # 26 letters + apostrophe + space + blank

    def text_to_int(self, text: str) -> list:
        """Map text string to an integer sequence."""
        int_sequence = []
        for c in text.upper():
            idx = self.char_map.get(c)
            if idx is not None:
                int_sequence.append(idx)
        return int_sequence

    def int_to_text(self, labels: list) -> str:
        """Map integer sequence to text string."""
        chars = []
        for i in labels:
            if i == self.blank_id:
                continue
            char = self.index_map.get(i, '')
            if char:
                chars.append(char)
        return ''.join(chars)


# Global cached TextTransform instance
_cached_text_transform = None


def get_text_transform() -> TextTransform:
    """Get cached TextTransform instance."""
    global _cached_text_transform
    if _cached_text_transform is None:
        _cached_text_transform = TextTransform()
    return _cached_text_transform


class TransformerLrScheduler:
    """
    Transformer LR scheduler from "Attention is all you need."

    References:
    - https://arxiv.org/abs/1706.03762
    - https://arxiv.org/abs/2005.08100 (Conformer paper)
    """

    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int,
        multiplier: float = 5.0,
    ):
        self._optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.n_steps = 0
        self.multiplier = multiplier

    def step(self):
        """Update learning rate."""
        self.n_steps += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self) -> float:
        """Compute learning rate."""
        return self.multiplier * (self.d_model ** -0.5) * min(
            self.n_steps ** (-0.5),
            self.n_steps * (self.warmup_steps ** (-1.5))
        )

    def get_last_lr(self) -> list:
        """Return last computed learning rate."""
        return [self._get_lr()]


class AvgMeter:
    """Running average meter for metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt if self.cnt > 0 else 0.0


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file does not exist: {config_path}')

    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: dict, config_path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_device_info() -> dict:
    """
    Detect available devices and return device information.
    Supports NVIDIA CUDA, AMD ROCm, and CPU.
    Auto-detects the best available accelerator.
    """
    device_info = {
        'device': 'cpu',
        'device_type': 'cpu',  # For autocast: 'cuda' or 'cpu'
        'device_name': 'CPU',
        'num_devices': 0,
        'is_cuda': False,
        'is_rocm': False,
        'use_distributed': False,
        'supports_amp': False,  # Mixed precision support
    }

    if torch.cuda.is_available():
        device_info['num_devices'] = torch.cuda.device_count()
        device_info['is_cuda'] = True
        device_info['device'] = 'cuda'
        device_info['device_type'] = 'cuda'  # ROCm uses 'cuda' device_type too
        device_info['supports_amp'] = True

        try:
            device_name = torch.cuda.get_device_name(0)
            if 'AMD' in device_name or 'Radeon' in device_name or 'MI' in device_name:
                device_info['is_rocm'] = True
                device_info['device_name'] = f'AMD GPU (ROCm): {device_name}'
            else:
                device_info['device_name'] = f'NVIDIA GPU (CUDA): {device_name}'
        except Exception:
            device_info['device_name'] = 'GPU (unknown type)'

        if device_info['num_devices'] > 1:
            device_info['use_distributed'] = True
            print(f"[Device] Detected {device_info['num_devices']} GPUs - Multi-GPU training available")
        else:
            print(f"[Device] {device_info['device_name']}")
    else:
        # CPU mode - check for torch.compile compatibility
        print("[Device] No GPU detected, using CPU")
        # CPU AMP is supported in PyTorch 1.10+ for bfloat16
        device_info['supports_amp'] = hasattr(torch.cpu.amp, 'autocast') if hasattr(torch, 'cpu') else False

    return device_info


def get_autocast_context(device_info: dict, enabled: bool = True):
    """
    Get the appropriate autocast context for the detected device.
    Works with NVIDIA CUDA, AMD ROCm, and CPU.

    Args:
        device_info: Device info from get_device_info()
        enabled: Whether to enable mixed precision

    Returns:
        Autocast context manager
    """
    from torch.amp import autocast

    if not enabled or not device_info.get('supports_amp', False):
        # Return a no-op context
        from contextlib import nullcontext
        return nullcontext()

    device_type = device_info.get('device_type', 'cpu')

    # For CUDA/ROCm, use float16; for CPU use bfloat16
    if device_type == 'cuda':
        return autocast(device_type='cuda', dtype=torch.float16)
    else:
        return autocast(device_type='cpu', dtype=torch.bfloat16)


def get_grad_scaler(device_info: dict, enabled: bool = True):
    """
    Get GradScaler for mixed precision training.
    GradScaler is only useful for CUDA/ROCm, not CPU.

    Args:
        device_info: Device info from get_device_info()
        enabled: Whether to enable gradient scaling

    Returns:
        GradScaler instance (may be disabled for CPU)
    """
    from torch.amp import GradScaler

    # GradScaler is only effective for CUDA devices
    use_scaler = enabled and device_info.get('device_type') == 'cuda'

    return GradScaler(enabled=use_scaler)


def setup_distributed(rank: int, world_size: int):
    """
    Setup distributed training environment.
    Works with both NVIDIA NCCL and AMD RCCL backends.
    """
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    return rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def clear_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_model_noise(model: nn.Module, std: float = 0.0001, device=None):
    """
    Add variational noise to model weights.
    Reference: https://ieeexplore.ieee.org/abstract/document/548170
    """
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.size(), device=device) * std
            param.add_(noise)


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    d_model: int,
) -> dict:
    """
    Estimate memory usage for a model.
    Useful for automatic batch size adjustment.
    """
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_memory = param_memory
    activation_memory = batch_size * seq_length * d_model * 4 * 10
    optimizer_memory = param_memory * 2

    total_memory = param_memory + grad_memory + activation_memory + optimizer_memory

    return {
        'parameters_mb': param_memory / (1024 ** 2),
        'gradients_mb': grad_memory / (1024 ** 2),
        'activations_mb': activation_memory / (1024 ** 2),
        'optimizer_mb': optimizer_memory / (1024 ** 2),
        'total_mb': total_memory / (1024 ** 2),
    }


def view_spectrogram(sample: torch.Tensor):
    """View spectrogram using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization")
        return

    specgram = sample.transpose(1, 0)
    plt.figure(figsize=(12, 4))
    plt.imshow(specgram.log2().detach().cpu().numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(label='Log Power')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Debug Utilities for Tensor Inspection
# =============================================================================

class TensorDebugger:
    """
    Comprehensive tensor debugger for tracing NaN/Inf issues.

    Usage:
        debugger = TensorDebugger(enabled=True, log_file="debug.log")
        debugger.check(tensor, "layer_name")
    """

    def __init__(
        self,
        enabled: bool = False,
        log_file: str = None,
        print_stats: bool = True,
        raise_on_nan: bool = False,
        max_history: int = 100,
    ):
        self.enabled = enabled
        self.log_file = log_file
        self.print_stats = print_stats
        self.raise_on_nan = raise_on_nan
        self.max_history = max_history

        self.history = []
        self.nan_locations = []
        self._file_handle = None

        if log_file and enabled:
            self._file_handle = open(log_file, 'w')
            self._log("=" * 80)
            self._log("TENSOR DEBUGGER INITIALIZED")
            self._log("=" * 80)

    def _log(self, msg: str):
        """Log message to file and/or console."""
        if self._file_handle:
            self._file_handle.write(msg + "\n")
            self._file_handle.flush()
        if self.print_stats:
            print(msg)

    def get_tensor_stats(self, tensor: torch.Tensor) -> dict:
        """Get comprehensive statistics for a tensor."""
        if tensor is None:
            return {"error": "tensor is None"}

        with torch.no_grad():
            t = tensor.float().detach()

            stats = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "has_nan": bool(torch.isnan(t).any()),
                "has_inf": bool(torch.isinf(t).any()),
                "nan_count": int(torch.isnan(t).sum()),
                "inf_count": int(torch.isinf(t).sum()),
            }

            # Compute stats only on finite values
            finite_mask = torch.isfinite(t)
            if finite_mask.any():
                finite_vals = t[finite_mask]
                stats.update({
                    "min": float(finite_vals.min()),
                    "max": float(finite_vals.max()),
                    "mean": float(finite_vals.mean()),
                    "std": float(finite_vals.std()) if finite_vals.numel() > 1 else 0.0,
                    "abs_max": float(finite_vals.abs().max()),
                })
            else:
                stats.update({
                    "min": float('nan'),
                    "max": float('nan'),
                    "mean": float('nan'),
                    "std": float('nan'),
                    "abs_max": float('nan'),
                })

            return stats

    def format_stats(self, name: str, stats: dict) -> str:
        """Format tensor stats as a readable string."""
        if "error" in stats:
            return f"[DEBUG] {name}: {stats['error']}"

        status = "OK"
        if stats["has_nan"]:
            status = f"NaN({stats['nan_count']})"
        elif stats["has_inf"]:
            status = f"Inf({stats['inf_count']})"

        return (
            f"[DEBUG] {name:40s} | "
            f"shape={stats['shape']} | "
            f"min={stats['min']:+.4e} max={stats['max']:+.4e} | "
            f"mean={stats['mean']:+.4e} std={stats['std']:.4e} | "
            f"abs_max={stats['abs_max']:.4e} | "
            f"{status}"
        )

    def check(
        self,
        tensor: torch.Tensor,
        name: str,
        step: int = None,
    ) -> dict:
        """
        Check tensor for issues and log statistics.

        Args:
            tensor: Tensor to check
            name: Name/label for this tensor
            step: Optional step/batch number

        Returns:
            Dictionary of tensor statistics
        """
        if not self.enabled:
            return {}

        stats = self.get_tensor_stats(tensor)
        stats["name"] = name
        stats["step"] = step

        # Log
        msg = self.format_stats(name, stats)
        if step is not None:
            msg = f"[Step {step}] " + msg
        self._log(msg)

        # Track history
        self.history.append(stats)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Track NaN locations
        if stats.get("has_nan") or stats.get("has_inf"):
            self.nan_locations.append({
                "name": name,
                "step": step,
                "nan_count": stats.get("nan_count", 0),
                "inf_count": stats.get("inf_count", 0),
            })

            if self.raise_on_nan:
                raise RuntimeError(f"NaN/Inf detected in {name} at step {step}")

        return stats

    def check_model_params(self, model: nn.Module, step: int = None):
        """Check all model parameters for NaN/Inf."""
        if not self.enabled:
            return

        self._log(f"\n{'='*80}")
        self._log(f"MODEL PARAMETER CHECK (step={step})")
        self._log("=" * 80)

        for name, param in model.named_parameters():
            self.check(param.data, f"param.{name}", step)
            if param.grad is not None:
                self.check(param.grad, f"grad.{name}", step)

    def check_grads(self, model: nn.Module, step: int = None):
        """Check gradients for NaN/Inf."""
        if not self.enabled:
            return

        self._log(f"\n--- Gradient Check (step={step}) ---")

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.check(param.grad, f"grad.{name}", step)

    def summary(self) -> str:
        """Get summary of all NaN/Inf occurrences."""
        if not self.nan_locations:
            return "No NaN/Inf detected"

        lines = [
            "\n" + "=" * 80,
            "NaN/Inf SUMMARY",
            "=" * 80,
        ]

        for loc in self.nan_locations:
            lines.append(
                f"  [{loc['step']}] {loc['name']}: "
                f"NaN={loc['nan_count']}, Inf={loc['inf_count']}"
            )

        return "\n".join(lines)

    def close(self):
        """Close log file."""
        if self._file_handle:
            self._log(self.summary())
            self._file_handle.close()


def debug_tensor(
    tensor: torch.Tensor,
    name: str = "tensor",
    verbose: bool = True,
) -> dict:
    """
    Quick debug function for a single tensor.

    Usage:
        debug_tensor(encoder_out, "encoder_out")
    """
    debugger = TensorDebugger(enabled=True, print_stats=verbose)
    return debugger.check(tensor, name)


def check_for_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Quick check if tensor has NaN or Inf values."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        print(f"[WARNING] {name} has {'NaN' if has_nan else ''}{' and ' if has_nan and has_inf else ''}{'Inf' if has_inf else ''}")
        return True
    return False


class DebugHook:
    """
    Forward/backward hook for debugging layer outputs.

    Usage:
        hook = DebugHook("encoder.layer_0")
        model.encoder.layers[0].register_forward_hook(hook.forward_hook)
        model.encoder.layers[0].register_full_backward_hook(hook.backward_hook)
    """

    def __init__(self, name: str, debugger: TensorDebugger = None):
        self.name = name
        self.debugger = debugger or TensorDebugger(enabled=True)
        self.step = 0

    def forward_hook(self, module, input, output):
        """Forward hook to check layer output."""
        if isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    self.debugger.check(o, f"{self.name}.output[{i}]", self.step)
        elif isinstance(output, torch.Tensor):
            self.debugger.check(output, f"{self.name}.output", self.step)

    def backward_hook(self, module, grad_input, grad_output):
        """Backward hook to check gradients."""
        if isinstance(grad_output, tuple):
            for i, g in enumerate(grad_output):
                if isinstance(g, torch.Tensor):
                    self.debugger.check(g, f"{self.name}.grad_out[{i}]", self.step)

    def set_step(self, step: int):
        self.step = step
