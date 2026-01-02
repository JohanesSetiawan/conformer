# Conformer ASR

PyTorch implementation of [Conformer](https://arxiv.org/abs/2005.08100) model for end-to-end speech recognition on the LibriSpeech dataset.

## Features

- JSON-based configuration system
- Multi-GPU support (auto-detected)
- AMD ROCm GPU support (MI300X, etc.)
- NVIDIA CUDA GPU support
- Mixed precision training (FP16/BF16)
- Memory-efficient training with gradient accumulation
- Smart batching for optimal performance
- Gradient clipping for training stability

## Installation

### For NVIDIA GPUs:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu<VERSION>
pip install torchmetrics
```

### For AMD ROCm GPUs:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm<VERSION>
pip install torchmetrics
```

## Usage

### Train with a config file:
```bash
python train.py --config config/conformer_small.json
```

### Available configurations:

| Config File | Model Size | Parameters | Recommended GPU |
|-------------|------------|------------|-----------------|
| `config/conformer_ultra_small.json` | Ultra Small | ~2M | Any GPU / CPU |
| `config/conformer_small.json` | Small (S) | ~10M | 8GB+ VRAM |
| `config/conformer_medium.json` | Medium (M) | ~30M | 16GB+ VRAM |
| `config/conformer_large.json` | Large (L) | ~118M | 40GB+ VRAM |

### Example: Train Conformer Small
```bash
python train.py --config config/conformer_small.json
```

### Example: Train on AMD MI300X
```bash
# ROCm is automatically detected
python train.py --config config/conformer_medium.json
```

## Configuration

All training parameters are controlled via JSON config files. Example config structure:

```json
{
  "model": {
    "name": "conformer_small",
    "d_input": 80,
    "d_encoder": 144,
    "d_decoder": 320,
    "encoder_layers": 16,
    "decoder_layers": 1,
    "attention_heads": 4,
    "conv_kernel_size": 31,
    "feed_forward_expansion_factor": 4,
    "feed_forward_residual_factor": 0.5,
    "dropout": 0.1,
    "num_classes": 29
  },
  "training": {
    "epochs": 50,
    "batch_size": 16,
    "accumulate_iters": 2,
    "warmup_steps": 10000,
    "weight_decay": 1e-6,
    "variational_noise_std": 0.0001,
    "gradient_clip_value": 1.0,
    "use_amp": true,
    "report_freq": 100
  },
  "data": {
    "data_dir": "./data",
    "train_set": "train-clean-100",
    "test_set": "test-clean",
    "num_workers": 4,
    "smart_batch": true
  },
  "checkpoint": {
    "checkpoint_path": "checkpoints/conformer_small_best.pt",
    "save_dir": "checkpoints",
    "load_checkpoint": false
  },
  "hardware": {
    "distributed": "auto",
    "mixed_precision": true,
    "memory_efficient": true
  }
}
```

### Hardware Configuration Options

- `distributed`: `"auto"` (detect GPUs), `"single"` (force single GPU), `"multi"` (force multi-GPU)
- `mixed_precision`: Enable FP16/BF16 training
- `memory_efficient`: Enable memory optimizations

## Model Variants

| Variant | d_encoder | Layers | Attention Heads | Parameters |
|---------|-----------|--------|-----------------|------------|
| Ultra Small | 64 | 4 | 2 | ~2M |
| Small (S) | 144 | 16 | 4 | ~10M |
| Medium (M) | 256 | 16 | 4 | ~30M |
| Large (L) | 512 | 17 | 8 | ~118M |

## Memory Optimization Tips

1. **Reduce batch size**: Use smaller batch_size with larger accumulate_iters
2. **Disable smart batching**: Set `"smart_batch": false` for large datasets
3. **Use mixed precision**: Set `"use_amp": true`
4. **Reduce num_workers**: Lower values use less RAM

## Supported Datasets

Uses torchaudio's [LibriSpeech dataset](https://pytorch.org/audio/stable/datasets.html):
- `train-clean-100`
- `train-clean-360`
- `train-other-500`
- `dev-clean`
- `dev-other`
- `test-clean`
- `test-other`

## Other Implementations
- https://github.com/sooftware/conformer
- https://github.com/lucidrains/conformer

## TODO
- Language Model (LM) implementation
- Support for other decoders (transformer decoder, etc.)
- Inference script for production use
