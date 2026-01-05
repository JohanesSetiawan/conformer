# FastConformer-Transducer ASR

A streaming-capable, memory-efficient Automatic Speech Recognition (ASR) model based on FastConformer encoder and Transducer decoder.

## Architecture

This implementation is based on NVIDIA's FastConformer research with the following key features:

- **FastConformer Encoder**: 8x subsampling for efficient attention computation
- **Depthwise Separable Convolutions**: Reduced compute with similar accuracy
- **Stateless Predictor**: Conv1D-based predictor (no LSTM state management)
- **RNN-T Loss**: Transducer-based training for streaming capability
- **Streaming Support**: Real-time transcription with configurable latency

### Key Optimizations

| Feature | Original Conformer | FastConformer |
|---------|-------------------|---------------|
| Subsampling | 4x | 8x |
| Conv Kernel | 31 | 9 |
| Convolutions | Standard | Depthwise Separable |
| Predictor | LSTM | Stateless Conv1D |

## Requirements

### System Dependencies

FFmpeg is required for audio loading:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Python Dependencies

```bash
# For NVIDIA GPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu<VERSION>
pip install -r requirements.txt

# For AMD GPU (ROCm)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm<VERSION>
pip install -r requirements.txt

# For CPU only
pip install torch torchaudio
pip install -r requirements.txt

# You can check on PyTorch Locally installation page for the latest version.
```

## Usage

### Training

```bash
# Train with small config (LibriSpeech train-clean-100)
python train.py --config config/fastconformer_small.json

# Resume training from checkpoint
python train.py --config config/fastconformer_small.json --resume

# Train with medium config (LibriSpeech train-clean-360)
python train.py --config config/fastconformer_medium.json

# Train with LJSpeech dataset
python train.py --config config/fastconformer_ljspeech.json
```

### Inference

```bash
# Offline transcription (full-context)
python inference.py --config config/fastconformer_small.json --audio audio.wav

# Streaming transcription (real-time)
python inference.py --config config/fastconformer_small.json --audio audio.wav --streaming

# With beam search decoding
python inference.py --config config/fastconformer_small.json --audio audio.wav --beam-size 5

# Show partial results during streaming
python inference.py --config config/fastconformer_small.json --audio audio.wav --streaming --show-partial
```

## Configuration

All configurations are JSON-based and located in the `config/` folder:

| Config | Encoder Dim | Layers | Params | Dataset |
|--------|-------------|--------|--------|---------|
| `fastconformer_small.json` | 256 | 12 | ~20M | LibriSpeech 100h |
| `fastconformer_medium.json` | 384 | 16 | ~50M | LibriSpeech 360h |
| `fastconformer_large.json` | 512 | 18 | ~90M | LibriSpeech 500h |
| `fastconformer_ljspeech.json` | 256 | 12 | ~20M | LJSpeech |

### Configuration Options

```json
{
  "model": {
    "d_input": 80,           // Mel spectrogram features
    "vocab_size": 29,        // A-Z + apostrophe + space + blank
    "d_encoder": 256,        // Encoder hidden dimension
    "d_predictor": 256,      // Predictor hidden dimension
    "d_joint": 320,          // Joint network dimension
    "encoder_layers": 12,    // Number of encoder layers
    "encoder_heads": 4,      // Attention heads
    "encoder_conv_kernel": 9,// Convolution kernel size
    "dropout": 0.1           // Dropout rate
  },
  "data": {
    "dataset": "librispeech", // or "ljspeech"
    "sample_rate": 16000
  },
  "training": {
    "epochs": 100,
    "batch_size": 8,
    "warmup_steps": 10000,
    "grad_clip": 1.0,
    "use_amp": true           // Mixed precision training
  },
  "wandb": {
    "enabled": false,         // Set to true to enable logging
    "project": "fastconformer-transducer"
  }
}
```

## Features

### Multi-GPU Training

Multi-GPU training is automatically enabled when multiple GPUs are detected:

```bash
# Uses all available GPUs automatically
python train.py --config config/fastconformer_medium.json
```

### AMD ROCm Support

Full support for AMD GPUs via ROCm. Install PyTorch with ROCm backend:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm<VERSION>
```

### Weights & Biases Logging

Enable W&B logging in the config:

```json
{
  "wandb": {
    "enabled": true,
    "project": "your-project-name",
    "run_name": "experiment-1"
  }
}
```

### Checkpointing

The training script maintains two checkpoints:
- `checkpoint.pt`: Latest checkpoint for resuming training
- `best_model.pt`: Best model based on validation WER

## Model Variants

### Small (~20M parameters)
- Suitable for edge devices and real-time applications
- ~50ms inference latency on modern GPUs
- Target WER: ~10% on LibriSpeech test-clean

### Medium (~50M parameters)
- Balanced accuracy and speed
- Good for server deployments
- Target WER: ~6% on LibriSpeech test-clean

### Large (~90M parameters)
- Maximum accuracy
- Best for offline transcription
- Target WER: ~4% on LibriSpeech test-clean

## Streaming Mode

The streaming mode processes audio in chunks with configurable latency:

```python
from inference import StreamingTranscriber
from model import create_model
from utils import load_config, get_text_transform

config = load_config('config/fastconformer_small.json')
model = create_model(config)
# Load checkpoint...

transcriber = StreamingTranscriber(
    model,
    get_text_transform(),
    device,
    chunk_size_ms=160,  # 160ms chunks
    lookahead_ms=80,    # 80ms lookahead
)

# Process audio chunks in real-time
for chunk in audio_stream:
    text = transcriber.process_chunk(chunk)
    if text:
        print(text, end='', flush=True)

# Finalize
final = transcriber.finalize()
```

## Supported Datasets

- **LibriSpeech**: train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other
- **LJSpeech**: Single speaker audiobook dataset

## References

- [FastConformer (NVIDIA)](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-06-07-fast-conformer/)
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [RNN Transducer](https://arxiv.org/abs/1211.3711)

## License

MIT License - See [LICENSE](LICENSE) for details.
