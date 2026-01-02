import torchaudio
import torch
import torch.nn as nn
import os
import random
import json
from functools import lru_cache

class TextTransform:
  ''' Map characters to integers and vice versa '''
  def __init__(self):
    self.char_map = {}
    for i, char in enumerate(range(65, 91)):
      self.char_map[chr(char)] = i
    self.char_map["'"] = 26
    self.char_map[' '] = 27
    self.index_map = {}
    for char, i in self.char_map.items():
      self.index_map[i] = char

  def text_to_int(self, text):
      ''' Map text string to an integer sequence '''
      int_sequence = []
      for c in text:
        ch = self.char_map.get(c)
        if ch is not None:
          int_sequence.append(ch)
      return int_sequence

  def int_to_text(self, labels):
      ''' Map integer sequence to text string '''
      string = []
      for i in labels:
          if i == 28: # blank char
            continue
          else:
            string.append(self.index_map.get(i, ''))
      return ''.join(string)


# Global cache for transforms to avoid recreating them every batch
# FIX: This prevents RAM spiking from recreating transforms
_cached_transforms = {}


def get_audio_transforms(cache_key='default'):
  '''
  Get audio transforms with caching to prevent RAM spiking.
  Transforms are created once and reused.
  '''
  global _cached_transforms

  if cache_key not in _cached_transforms:
    # 10 time masks with p=0.05
    # The actual conformer paper uses a variable time_mask_param based on the length of each utterance.
    # For simplicity, we approximate it with just a fixed value.
    time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
    train_audio_transform = nn.Sequential(
      torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160), # 80 filter banks, 25ms window size, 10ms hop
      torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
      *time_masks,
    )

    valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)

    _cached_transforms[cache_key] = (train_audio_transform, valid_audio_transform)

  return _cached_transforms[cache_key]


# Global cached TextTransform instance
_cached_text_transform = None

def get_text_transform():
  '''Get cached TextTransform instance to prevent repeated creation'''
  global _cached_text_transform
  if _cached_text_transform is None:
    _cached_text_transform = TextTransform()
  return _cached_text_transform


class BatchSampler(object):
  '''
  Sample contiguous, sorted indices. Leads to less padding and faster training.
  Optimized to reduce memory usage.
  '''
  def __init__(self, sorted_inds, batch_size):
    self.sorted_inds = sorted_inds
    self.batch_size = batch_size

  def __iter__(self):
    # FIX: Use more memory-efficient approach
    inds = list(self.sorted_inds)  # Only copy when iterating
    while len(inds):
      to_take = min(self.batch_size, len(inds))
      start_ind = random.randint(0, len(inds) - to_take)
      batch_inds = inds[start_ind:start_ind + to_take]
      del inds[start_ind:start_ind + to_take]
      yield batch_inds

  def __len__(self):
    return (len(self.sorted_inds) + self.batch_size - 1) // self.batch_size


def preprocess_example(data, data_type="train"):
  '''
  Process raw LibriSpeech examples.
  Uses cached transforms and text_transform to prevent RAM spiking.
  '''
  # FIX: Use cached transforms instead of creating new ones every batch
  text_transform = get_text_transform()
  train_audio_transform, valid_audio_transform = get_audio_transforms()

  spectrograms = []
  labels = []
  references = []
  input_lengths = []
  label_lengths = []

  for (waveform, _, utterance, _, _, _) in data:
    # Generate spectrogram for model input
    if data_type == 'train':
      spec = train_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
    else:
      spec = valid_audio_transform(waveform).squeeze(0).transpose(0, 1) # (1, time, freq)
    spectrograms.append(spec)

    # Labels
    references.append(utterance) # Actual Sentence
    label = torch.Tensor(text_transform.text_to_int(utterance)) # Integer representation of sentence
    labels.append(label)

    # Lengths (time)
    input_lengths.append(((spec.shape[0] - 1) // 2 - 1) // 2) # account for subsampling of time dimension
    label_lengths.append(len(label))

  # Pad batch to length of longest sample
  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  # Padding mask (batch_size, time, time)
  mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
  for i, l in enumerate(input_lengths):
    mask[i, :, :l] = 0

  return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()


class TransformerLrScheduler():
  '''
    Transformer LR scheduler from "Attention is all you need." https://arxiv.org/abs/1706.03762
    multiplier and warmup_steps taken from conformer paper: https://arxiv.org/abs/2005.08100
  '''
  def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
    self._optimizer = optimizer
    self.d_model = d_model
    self.warmup_steps = warmup_steps
    self.n_steps = 0
    self.multiplier = multiplier

  def step(self):
    self.n_steps += 1
    lr = self._get_lr()
    for param_group in self._optimizer.param_groups:
        param_group['lr'] = lr

  def _get_lr(self):
    return self.multiplier * (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))

  def get_last_lr(self):
    '''Return last computed learning rate'''
    return [self._get_lr()]


def model_size(model, name):
  ''' Print model size in num_params and MB'''
  param_size = 0
  num_params = 0
  for param in model.parameters():
    num_params += param.nelement()
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    num_params += buffer.nelement()
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print(f'{name} - num_params: {round(num_params / 1000000, 2)}M,  size: {round(size_all_mb, 2)}MB')


class GreedyCharacterDecoder(nn.Module):
  ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
  def __init__(self):
    super(GreedyCharacterDecoder, self).__init__()

  def forward(self, x):
    indices = torch.argmax(x, dim=-1)
    indices = torch.unique_consecutive(indices, dim=-1)
    return indices.tolist()


class AvgMeter(object):
  '''
    Keep running average for a metric
  '''
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = None
    self.sum = None
    self.cnt = 0

  def update(self, val, n=1):
    if not self.sum:
      self.sum = val * n
    else:
      self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def view_spectrogram(sample):
  ''' View spectrogram '''
  specgram = sample.transpose(1, 0)
  import matplotlib.pyplot as plt
  plt.figure()
  p = plt.imshow(specgram.log2()[:,:].detach().numpy(), cmap='gray')
  plt.show()


def add_model_noise(model, std=0.0001, device=None):
  '''
    Add variational noise to model weights: https://ieeexplore.ieee.org/abstract/document/548170
    STD may need some fine tuning...

    FIX: Now correctly uses the device parameter instead of always using cuda
  '''
  if device is None:
    # Auto-detect device from model parameters
    device = next(model.parameters()).device

  with torch.no_grad():
    for param in model.parameters():
      # FIX: Create noise on the same device as the parameter
      noise = torch.randn(param.size(), device=device) * std
      param.add_(noise)


def load_checkpoint(encoder, decoder, optimizer, scheduler, checkpoint_path):
  '''
  Load model checkpoint

  FIX: Proper exception handling
  '''
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')

  checkpoint = torch.load(checkpoint_path, map_location='cpu')
  scheduler.n_steps = checkpoint['scheduler_n_steps']
  scheduler.multiplier = checkpoint['scheduler_multiplier']
  scheduler.warmup_steps = checkpoint['scheduler_warmup_steps']
  encoder.load_state_dict(checkpoint['encoder_state_dict'])
  decoder.load_state_dict(checkpoint['decoder_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return checkpoint['epoch'], checkpoint['valid_loss']


def save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch, checkpoint_path):
  ''' Save model checkpoint '''
  # Ensure directory exists
  checkpoint_dir = os.path.dirname(checkpoint_path)
  if checkpoint_dir and not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

  torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'scheduler_n_steps': scheduler.n_steps,
            'scheduler_multiplier': scheduler.multiplier,
            'scheduler_warmup_steps': scheduler.warmup_steps,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)


def load_config(config_path):
  '''Load configuration from JSON file'''
  if not os.path.exists(config_path):
    raise FileNotFoundError(f'Config file does not exist: {config_path}')

  with open(config_path, 'r') as f:
    config = json.load(f)
  return config


def get_device_info():
  '''
  Detect available devices and return device information.
  Supports NVIDIA CUDA and AMD ROCm.
  '''
  device_info = {
    'device': 'cpu',
    'device_name': 'CPU',
    'num_devices': 0,
    'is_cuda': False,
    'is_rocm': False,
    'use_distributed': False,
  }

  # Check for CUDA (NVIDIA) or ROCm (AMD)
  if torch.cuda.is_available():
    device_info['num_devices'] = torch.cuda.device_count()
    device_info['is_cuda'] = True

    # Check if this is ROCm (AMD)
    # ROCm uses the same torch.cuda API but with different backend
    try:
      # Try to detect ROCm by checking the device name
      device_name = torch.cuda.get_device_name(0)
      if 'AMD' in device_name or 'Radeon' in device_name or 'MI' in device_name:
        device_info['is_rocm'] = True
        device_info['device_name'] = f'AMD GPU: {device_name}'
      else:
        device_info['device_name'] = f'NVIDIA GPU: {device_name}'
    except Exception:
      device_info['device_name'] = 'GPU (unknown type)'

    device_info['device'] = 'cuda'

    if device_info['num_devices'] > 1:
      device_info['use_distributed'] = True
      print(f"Detected {device_info['num_devices']} GPUs - Multi-GPU training enabled")
    else:
      print(f"Detected 1 GPU: {device_info['device_name']}")

  else:
    print("No GPU detected, using CPU")

  return device_info


def setup_distributed(rank, world_size):
  '''
  Setup distributed training environment.
  Works with both NVIDIA NCCL and AMD RCCL backends.
  '''
  import torch.distributed as dist

  # Detect backend - NCCL for NVIDIA, RCCL for AMD (uses same 'nccl' string)
  backend = 'nccl' if torch.cuda.is_available() else 'gloo'

  os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
  os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

  dist.init_process_group(backend, rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

  return rank


def cleanup_distributed():
  '''Cleanup distributed training'''
  import torch.distributed as dist
  if dist.is_initialized():
    dist.destroy_process_group()


def get_sorted_indices_lazy(dataset, max_samples=None):
  '''
  Lazily compute sorted indices for smart batching.
  FIX: More memory efficient - doesn't load all audio into memory at once.

  Args:
    dataset: The dataset to sort
    max_samples: Maximum number of samples to process (for large datasets)
  '''
  print('Computing audio lengths for smart batching (memory efficient)...')

  lengths = []
  total_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

  for i in range(total_samples):
    try:
      # Only access the waveform to get its length, don't process it
      waveform, *_ = dataset[i]
      lengths.append((i, waveform.shape[1]))
    except Exception as e:
      print(f"Warning: Could not process sample {i}: {e}")
      lengths.append((i, 0))

    # Progress indicator for large datasets
    if (i + 1) % 10000 == 0:
      print(f'  Processed {i + 1}/{total_samples} samples...')

  # Sort by length
  lengths.sort(key=lambda x: x[1])
  sorted_indices = [idx for idx, _ in lengths]

  print(f'Finished sorting {len(sorted_indices)} samples')
  return sorted_indices


class MemoryEfficientBatchSampler:
  '''
  Memory-efficient batch sampler that doesn't require pre-sorting the entire dataset.
  Groups samples by approximate length buckets.
  '''
  def __init__(self, dataset, batch_size, num_buckets=10):
    self.dataset = dataset
    self.batch_size = batch_size
    self.num_buckets = num_buckets
    self.length = len(dataset)

    # Create buckets based on index (approximate length grouping)
    self.bucket_size = self.length // num_buckets

  def __iter__(self):
    # Create random bucket order
    bucket_order = list(range(self.num_buckets))
    random.shuffle(bucket_order)

    for bucket_idx in bucket_order:
      start = bucket_idx * self.bucket_size
      end = min(start + self.bucket_size, self.length)

      indices = list(range(start, end))
      random.shuffle(indices)

      for i in range(0, len(indices), self.batch_size):
        batch = indices[i:i + self.batch_size]
        if batch:
          yield batch

  def __len__(self):
    return (self.length + self.batch_size - 1) // self.batch_size


def estimate_memory_usage(model, batch_size, seq_length, d_model):
  '''
  Estimate memory usage for a model.
  Useful for automatic batch size adjustment.
  '''
  # Model parameters
  param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

  # Gradients (same size as parameters)
  grad_memory = param_memory

  # Activations (rough estimate)
  # Assuming 4 bytes per float32 element
  activation_memory = batch_size * seq_length * d_model * 4 * 10  # 10 for multiple layers

  # Optimizer states (Adam uses 2x param memory for momentum and variance)
  optimizer_memory = param_memory * 2

  total_memory = param_memory + grad_memory + activation_memory + optimizer_memory

  return {
    'parameters_mb': param_memory / (1024 ** 2),
    'gradients_mb': grad_memory / (1024 ** 2),
    'activations_mb': activation_memory / (1024 ** 2),
    'optimizer_mb': optimizer_memory / (1024 ** 2),
    'total_mb': total_memory / (1024 ** 2),
  }


def clear_memory():
  '''Clear GPU memory cache and run garbage collection'''
  import gc
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
