"""
FastConformer-Transducer Training Script

Features:
- RNN-T loss training
- tqdm progress bars
- Weights & Biases logging (optional)
- Dual checkpointing (resume + best WER)
- Multi-GPU training (DDP)
- Mixed precision training (AMP)
- Support for LibriSpeech and LJSpeech datasets
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# Note: autocast and GradScaler are handled via utils.get_autocast_context and utils.get_grad_scaler
import torchaudio
import torchaudio.functional as F_audio
from tqdm import tqdm
from torchmetrics.text import WordErrorRate, CharErrorRate

from model import FastConformerTransducer, create_model, model_size
from utils import (
    get_device_info,
    get_autocast_context,
    get_grad_scaler,
    setup_distributed,
    cleanup_distributed,
    load_config,
    TransformerLrScheduler,
    clear_memory,
    get_text_transform,
    AvgMeter,
    TensorDebugger,
    check_for_nan,
)


def get_audio_transforms(sample_rate: int = 16000, n_mels: int = 80, training: bool = True):
    """Get audio transforms for mel spectrogram generation with SpecAugment."""
    hop_length = sample_rate // 100  # 10ms hop

    if training:
        # Training with SpecAugment
        time_masks = [
            torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05)
            for _ in range(10)
        ]
        transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                hop_length=hop_length,
            ),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
            *time_masks,
        )
    else:
        # Validation without augmentation
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
        )

    return transform


class TransducerDataset(torch.utils.data.Dataset):
    """Dataset wrapper for Transducer training."""

    def __init__(
        self,
        dataset,
        text_transform,
        audio_transform,
        target_sample_rate: int = 16000,
        dataset_type: str = "librispeech",
    ):
        self.dataset = dataset
        self.text_transform = text_transform
        self.audio_transform = audio_transform
        self.target_sample_rate = target_sample_rate
        self.dataset_type = dataset_type

        # Resampler cache
        self._resamplers = {}

    def __len__(self):
        return len(self.dataset)

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        """Get cached resampler for a given sample rate."""
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr,
                new_freq=self.target_sample_rate,
            )
        return self._resamplers[orig_sr]

    def __getitem__(self, idx):
        if self.dataset_type == "librispeech":
            waveform, sample_rate, utterance, _, _, _ = self.dataset[idx]
        elif self.dataset_type == "ljspeech":
            waveform, sample_rate, _, utterance = self.dataset[idx]
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        # Convert to mel spectrogram
        mel = self.audio_transform(waveform).squeeze(0).transpose(0, 1)  # (time, n_mels)

        # Convert text to token indices
        tokens = torch.tensor(
            self.text_transform.text_to_int(utterance.upper()),
            dtype=torch.long,
        )

        return mel, tokens, utterance


def collate_fn(batch, subsampling_factor: int = 8):
    """Collate function for DataLoader with RNN-T length validation.

    RNN-T loss requires encoder_length >= token_length for all samples.
    This function filters out samples that violate this constraint.

    Args:
        batch: List of (mel, tokens, utterance) tuples
        subsampling_factor: Encoder subsampling factor (default 8 for FastConformer
            with 3 conv layers @ stride 2 each = 2*2*2 = 8x subsampling)

    Returns:
        Tuple of (mels_padded, mel_lengths, tokens_padded, token_lengths, utterances)
        Returns None if all samples are filtered out
    """
    mels, tokens, utterances = zip(*batch)

    # Calculate expected encoder lengths after subsampling
    mel_lengths = [m.size(0) for m in mels]
    token_lengths = [t.size(0) for t in tokens]

    # Filter samples that satisfy RNN-T constraint: encoder_length >= token_length
    # Use conservative estimate with margin for conv layer edge effects
    valid_indices = []
    for i, (mel_len, tok_len) in enumerate(zip(mel_lengths, token_lengths)):
        # Conservative encoder length estimate with margin for conv edge effects
        # Subtract 2 for conv kernel edge effects to be safe
        encoder_len = max(1, (mel_len // subsampling_factor) - 2)
        if encoder_len >= tok_len:
            valid_indices.append(i)

    # If no valid samples, return None (will be handled in training loop)
    if len(valid_indices) == 0:
        return None

    # Filter to valid samples only
    if len(valid_indices) < len(batch):
        mels = [mels[i] for i in valid_indices]
        tokens = [tokens[i] for i in valid_indices]
        utterances = [utterances[i] for i in valid_indices]

    # Get lengths before padding
    mel_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)
    token_lengths = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)

    # Pad sequences
    mels_padded = nn.utils.rnn.pad_sequence(mels, batch_first=True)
    tokens_padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    return mels_padded, mel_lengths, tokens_padded, token_lengths, utterances


def greedy_decode(model, encoder_out, max_symbols: int = 100):
    """Greedy decoding for Transducer model."""
    batch_size, T, _ = encoder_out.size()
    device = encoder_out.device

    # Initialize
    blank_id = model.blank_id
    decoded = [[] for _ in range(batch_size)]

    for b in range(batch_size):
        # Start with blank
        y = torch.zeros(1, 1, dtype=torch.long, device=device)
        predictor_out = model.predict(y)

        for t in range(T):
            enc_t = encoder_out[b:b+1, t:t+1, :]  # (1, 1, d_encoder)

            for _ in range(max_symbols):
                logits = model.joint_step(enc_t, predictor_out)  # (1, vocab_size)
                pred = logits.argmax(dim=-1).item()

                if pred == blank_id:
                    break

                decoded[b].append(pred)

                # Update predictor state
                y = torch.tensor([[pred]], dtype=torch.long, device=device)
                pred_out = model.predict(y)
                predictor_out = pred_out

    return decoded


def validate(model, val_loader, device, text_transform, rank=0):
    """Validate model and compute WER."""
    model.eval()

    wer_metric = WordErrorRate()
    cer_metric = CharErrorRate()

    all_predictions = []
    all_references = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", disable=(rank != 0))

        for batch in pbar:
            mels, mel_lengths, tokens, token_lengths, utterances = batch
            mels = mels.to(device)
            mel_lengths = mel_lengths.to(device)

            # Encode
            encoder_out, encoder_lengths, _ = model.encode(mels, mel_lengths, mode="offline")

            # Greedy decode
            decoded = greedy_decode(model, encoder_out)

            # Convert to text
            for i, dec in enumerate(decoded):
                pred_text = text_transform.int_to_text(dec)
                ref_text = utterances[i].upper()

                all_predictions.append(pred_text)
                all_references.append(ref_text)

    # Compute metrics
    wer = wer_metric(all_predictions, all_references).item() * 100
    cer = cer_metric(all_predictions, all_references).item() * 100

    return wer, cer, all_predictions[:5], all_references[:5]


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    device,
    device_info,
    epoch,
    config,
    rank=0,
    wandb_run=None,
    debugger=None,
):
    """Train for one epoch."""
    model.train()

    loss_meter = AvgMeter()
    grad_norm_meter = AvgMeter()

    # Get blank_id from model (handle DDP wrapper)
    blank_id = model.module.blank_id if hasattr(model, 'module') else model.blank_id

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=(rank != 0),
    )

    use_amp = config['training'].get('use_amp', True)
    debug_mode = config['training'].get('debug', False)
    debug_interval = config['training'].get('debug_interval', 50)

    nan_count = 0
    skip_count = 0
    max_nan_skip = 10  # Skip up to 10 NaN batches before raising error

    for batch_idx, batch in enumerate(pbar):
        # Calculate debug flag early
        should_debug = debug_mode and (batch_idx % debug_interval == 0)

        # Handle None batch (all samples filtered due to RNN-T length constraint)
        if batch is None:
            skip_count += 1
            if rank == 0 and skip_count <= 5:
                print(f"\n[Info] Batch {batch_idx} skipped: all samples violated RNN-T length constraint")
            continue

        mels, mel_lengths, tokens, token_lengths, _ = batch

        # Log if samples were filtered
        if should_debug and rank == 0:
            original_batch_size = config['training'].get('batch_size', 8)
            if mels.size(0) < original_batch_size:
                print(f"[Debug] Batch {batch_idx}: {mels.size(0)}/{original_batch_size} samples (some filtered)")

        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)
        if should_debug and debugger:
            debugger.check(mels, "input.mels", batch_idx)
            debugger.check(tokens.float(), "input.tokens", batch_idx)

        optimizer.zero_grad()

        # Mixed precision forward (device-agnostic)
        with get_autocast_context(device_info, enabled=use_amp):
            # Use debug flag in model forward if debug mode is on
            logits, encoder_lengths = model(
                mels, mel_lengths, tokens, token_lengths,
                debug=should_debug,
            )

            # Debug logits before loss computation
            if should_debug and debugger:
                debugger.check(logits, "forward.logits", batch_idx)
                debugger.check(encoder_lengths.float(), "forward.encoder_lengths", batch_idx)

            # Runtime validation: check RNN-T constraint after getting actual encoder_lengths
            # encoder_length must be >= token_length for all samples
            constraint_violated = (encoder_lengths < token_lengths).any().item()
            if constraint_violated:
                if rank == 0:
                    violations = (encoder_lengths < token_lengths).nonzero(as_tuple=True)[0].tolist()
                    print(f"\n[Warning] RNN-T constraint violated at batch {batch_idx}")
                    print(f"  encoder_lengths: {encoder_lengths.tolist()}")
                    print(f"  token_lengths: {token_lengths.tolist()}")
                    print(f"  violations at indices: {violations}")
                skip_count += 1
                optimizer.zero_grad()
                continue

            # Prepare tensors for RNN-T loss (float32 on CPU)
            logits_cpu = logits.float().cpu()
            tokens_cpu = tokens.int().cpu()
            enc_len_cpu = encoder_lengths.int().cpu()
            tok_len_cpu = token_lengths.int().cpu()

            # Debug loss inputs
            if should_debug and debugger:
                debugger.check(logits_cpu, "rnnt.logits_cpu", batch_idx)
                print(f"[DEBUG] rnnt.tokens_cpu: shape={list(tokens_cpu.shape)}, values={tokens_cpu[0, :10].tolist()}")
                print(f"[DEBUG] rnnt.enc_lengths: {enc_len_cpu.tolist()}")
                print(f"[DEBUG] rnnt.tok_lengths: {tok_len_cpu.tolist()}")

            # RNN-T loss (compute on CPU as torchaudio only supports CPU backend)
            # This works for both NVIDIA CUDA and AMD ROCm
            loss = F_audio.rnnt_loss(
                logits_cpu,
                tokens_cpu,
                enc_len_cpu,
                tok_len_cpu,
                blank=blank_id,
                reduction='mean',
            ).to(device)

            if should_debug:
                print(f"[DEBUG] loss: {loss.item():.6f}")

        # Check for NaN loss and skip batch if needed
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if rank == 0:
                print(f"\n[Warning] NaN/Inf loss at batch {batch_idx}")
                # Print detailed debug info on NaN
                print(f"  logits: min={logits.min().item():.4e}, max={logits.max().item():.4e}")
                print(f"  encoder_lengths: {encoder_lengths.tolist()}")
                print(f"  token_lengths: {token_lengths.tolist()}")
                if debugger:
                    debugger.check(logits, f"NaN.logits.batch{batch_idx}", batch_idx)

            if nan_count > max_nan_skip:
                if debugger:
                    print(debugger.summary())
                raise RuntimeError(f"Too many NaN losses ({nan_count}). Training unstable.")

            optimizer.zero_grad()
            continue

        # Backward with gradient scaling (scaler handles enabled/disabled automatically)
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training'].get('grad_clip', 1.0),
        )

        # Debug gradients
        if should_debug and debugger:
            debugger.check_grads(model.module if hasattr(model, 'module') else model, batch_idx)

        # Check for NaN gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            nan_count += 1
            if rank == 0:
                print(f"\n[Warning] NaN/Inf gradient at batch {batch_idx}, grad_norm={grad_norm}")

            if nan_count > max_nan_skip:
                if debugger:
                    print(debugger.summary())
                raise RuntimeError(f"Too many NaN gradients ({nan_count}). Training unstable.")

            # Reset scaler state and skip this batch
            # scaler.update() resets internal state after unscale_ was called
            optimizer.zero_grad()
            scaler.update()
            continue

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Reset NaN count on successful step
        nan_count = 0

        # Update meters
        loss_meter.update(loss.item())
        grad_norm_meter.update(grad_norm.item())

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{current_lr:.2e}',
            'grad': f'{grad_norm_meter.avg:.2f}',
        })

        # Log to wandb
        if wandb_run is not None and batch_idx % 100 == 0:
            wandb_run.log({
                'train/loss': loss.item(),
                'train/lr': current_lr,
                'train/grad_norm': grad_norm.item(),
                'train/step': scheduler.n_steps,
            })

    return loss_meter.avg


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    wer,
    config,
    checkpoint_path,
    is_best=False,
):
    """Save model checkpoint."""
    # Handle DDP wrapped models
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'wer': wer,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_n_steps': scheduler.n_steps,
        'scheduler_multiplier': scheduler.multiplier,
        'scheduler_warmup_steps': scheduler.warmup_steps,
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_path.replace('checkpoint.pt', 'best_model.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    checkpoint_path,
    device,
):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, float('inf')

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DDP wrapped models
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.n_steps = checkpoint['scheduler_n_steps']
    scheduler.multiplier = checkpoint['scheduler_multiplier']
    scheduler.warmup_steps = checkpoint['scheduler_warmup_steps']
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch'], checkpoint.get('wer', float('inf'))


def get_dataset(config, split: str = "train"):
    """Get dataset based on configuration."""
    dataset_config = config['data']
    dataset_name = dataset_config.get('dataset', 'librispeech')

    if dataset_name == 'librispeech':
        # LibriSpeech
        subset = dataset_config.get('train_subset', 'train-clean-100') if split == 'train' else dataset_config.get('val_subset', 'dev-clean')
        root = dataset_config.get('data_root', './data')

        # Create directory if not exists
        os.makedirs(root, exist_ok=True)

        dataset = torchaudio.datasets.LIBRISPEECH(
            root=root,
            url=subset,
            download=dataset_config.get('download', True),
        )
        return dataset, 'librispeech'

    elif dataset_name == 'ljspeech':
        # LJSpeech
        root = dataset_config.get('data_root', './data')

        # Create directory if not exists
        os.makedirs(root, exist_ok=True)

        full_dataset = torchaudio.datasets.LJSPEECH(
            root=root,
            download=dataset_config.get('download', True),
        )

        # Split LJSpeech (no official splits)
        n_total = len(full_dataset)
        n_val = int(n_total * 0.05)  # 5% for validation

        if split == 'train':
            return torch.utils.data.Subset(full_dataset, range(n_val, n_total)), 'ljspeech'
        else:
            return torch.utils.data.Subset(full_dataset, range(n_val)), 'ljspeech'

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train(rank, world_size, config, args, device_info):
    """Main training function."""
    # Setup distributed if multi-GPU
    is_distributed = world_size > 1
    if is_distributed:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device(device_info['device'])

    is_main = (rank == 0)

    if is_main:
        print(f"[Training] Device: {device_info['device_name']}")
        print(f"[Training] AMP enabled: {device_info['supports_amp']}")

    # Initialize wandb (main process only)
    wandb_run = None
    if is_main and config.get('wandb', {}).get('enabled', False):
        import wandb
        wandb_run = wandb.init(
            project=config['wandb'].get('project', 'fastconformer-transducer'),
            name=config['wandb'].get('run_name', None),
            config=config,
        )

    # Initialize debugger (main process only)
    debugger = None
    debug_mode = config['training'].get('debug', False)
    if is_main and debug_mode:
        checkpoint_dir = config['training'].get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        debug_log_file = os.path.join(checkpoint_dir, 'debug.log')
        debugger = TensorDebugger(
            enabled=True,
            log_file=debug_log_file,
            print_stats=True,
            raise_on_nan=False,
        )
        print(f"[Debug] Debug mode enabled, logging to {debug_log_file}")

    # Create model
    model = create_model(config)
    model = model.to(device)

    if is_main:
        model_size(model, "FastConformer-Transducer")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    # Text transform
    text_transform = get_text_transform()

    # Audio transforms
    train_audio_transform = get_audio_transforms(
        sample_rate=config['data'].get('sample_rate', 16000),
        n_mels=config['model'].get('d_input', 80),
        training=True,
    ).to(device)

    val_audio_transform = get_audio_transforms(
        sample_rate=config['data'].get('sample_rate', 16000),
        n_mels=config['model'].get('d_input', 80),
        training=False,
    ).to(device)

    # Datasets
    train_data, train_type = get_dataset(config, 'train')
    val_data, val_type = get_dataset(config, 'val')

    train_dataset = TransducerDataset(
        train_data,
        text_transform,
        train_audio_transform.cpu(),  # Move back to CPU for DataLoader workers
        target_sample_rate=config['data'].get('sample_rate', 16000),
        dataset_type=train_type,
    )

    val_dataset = TransducerDataset(
        val_data,
        text_transform,
        val_audio_transform.cpu(),
        target_sample_rate=config['data'].get('sample_rate', 16000),
        dataset_type=val_type,
    )

    # DataLoaders
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].get('batch_size', 8),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('batch_size', 8),
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['training'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training'].get('lr', 1e-3),
        weight_decay=config['training'].get('weight_decay', 1e-6),
        betas=(0.9, 0.98),
    )

    # Scheduler
    d_model = config['model'].get('d_encoder', 256)
    warmup_steps = config['training'].get('warmup_steps', 10000)
    scheduler = TransformerLrScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
        multiplier=config['training'].get('lr_multiplier', 5),
    )

    # Gradient scaler for mixed precision (device-agnostic)
    use_amp = config['training'].get('use_amp', True)
    scaler = get_grad_scaler(device_info, enabled=use_amp)

    # Load checkpoint if resuming
    start_epoch = 0
    best_wer = float('inf')
    checkpoint_dir = config['training'].get('checkpoint_dir', './checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')

    if args.resume and os.path.exists(checkpoint_path):
        start_epoch, best_wer = load_checkpoint(
            model, optimizer, scheduler, scaler, checkpoint_path, device
        )
        start_epoch += 1
        if is_main:
            print(f"Resumed from epoch {start_epoch}, best WER: {best_wer:.2f}%")

    # Training loop
    num_epochs = config['training'].get('epochs', 100)

    for epoch in range(start_epoch, num_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, device_info, epoch, config, rank, wandb_run, debugger
        )

        # Validate
        wer, cer, sample_preds, sample_refs = validate(
            model, val_loader, device, text_transform, rank
        )

        if is_main:
            # Print results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  WER: {wer:.2f}%")
            print(f"  CER: {cer:.2f}%")
            print(f"\nSample Predictions:")
            for i, (pred, ref) in enumerate(zip(sample_preds, sample_refs)):
                print(f"  [{i+1}] Ref: {ref}")
                print(f"       Pred: {pred}")

            # Log to wandb
            if wandb_run is not None:
                wandb_run.log({
                    'val/wer': wer,
                    'val/cer': cer,
                    'val/train_loss': train_loss,
                    'epoch': epoch,
                })

            # Save checkpoint
            is_best = wer < best_wer
            if is_best:
                best_wer = wer
                print(f"  New best WER: {best_wer:.2f}%")

            save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, wer, config, checkpoint_path, is_best
            )

        # Clear memory
        clear_memory()

    # Cleanup
    if is_distributed:
        cleanup_distributed()

    if wandb_run is not None:
        wandb_run.finish()

    # Close debugger
    if debugger is not None:
        debugger.close()


def main():
    parser = argparse.ArgumentParser(description='Train FastConformer-Transducer ASR')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get device info (auto-detect CPU/CUDA/ROCm)
    device_info = get_device_info()
    world_size = device_info['num_devices'] if device_info['use_distributed'] else 1

    print(f"\n{'='*60}")
    print("FastConformer-Transducer Training")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Device: {device_info['device_name']}")
    print(f"Distributed: {device_info['use_distributed']}")
    print(f"World size: {world_size}")
    print(f"{'='*60}\n")

    if world_size > 1:
        # Multi-GPU training
        mp.spawn(
            train,
            args=(world_size, config, args, device_info),
            nprocs=world_size,
            join=True,
        )
    else:
        # Single GPU or CPU training
        train(0, 1, config, args, device_info)


if __name__ == '__main__':
    main()
