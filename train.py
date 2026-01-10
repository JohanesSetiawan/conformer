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
    """Collate function for DataLoader with dynamic audio padding.

    Instead of filtering samples, we pad short audio to ensure
    encoder_length >= token_length (RNN-T requirement).

    Args:
        batch: List of (mel, tokens, utterance) tuples
        subsampling_factor: Encoder subsampling factor (default 8 for FastConformer)

    Returns:
        Tuple of (mels_padded, mel_lengths, tokens_padded, token_lengths, utterances)
    """
    mels, tokens, utterances = zip(*batch)

    # Pad each mel to ensure encoder_length >= token_length
    # encoder_length ≈ mel_length // subsampling_factor
    # So we need: mel_length >= token_length * subsampling_factor
    padded_mels = []
    mel_lengths = []
    for mel, tok in zip(mels, tokens):
        mel_len = mel.size(0)
        tok_len = tok.size(0)

        # Calculate minimum mel length needed (with margin for conv edges)
        min_mel_len = (tok_len + 3) * subsampling_factor  # +3 margin for safety

        if mel_len < min_mel_len:
            # Pad with zeros (silence)
            pad_len = min_mel_len - mel_len
            padding = torch.zeros(pad_len, mel.size(1), dtype=mel.dtype)
            mel = torch.cat([mel, padding], dim=0)

        padded_mels.append(mel)
        mel_lengths.append(mel.size(0))

    # Get lengths
    mel_lengths = torch.tensor(mel_lengths, dtype=torch.long)
    token_lengths = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)

    # Pad sequences to batch
    mels_padded = nn.utils.rnn.pad_sequence(padded_mels, batch_first=True)
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
            encoder_out, encoder_lengths, _ = model.encode(mels, mel_lengths)

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

    # Tracking counters
    nan_loss_count = 0
    nan_grad_count = 0
    total_batches = 0
    max_nan_skip = config['training'].get('max_nan_skip', 100)

    for batch_idx, batch in enumerate(pbar):
        total_batches += 1

        mels, mel_lengths, tokens, token_lengths, _ = batch
        mels = mels.to(device)
        mel_lengths = mel_lengths.to(device)
        tokens = tokens.to(device)
        token_lengths = token_lengths.to(device)

        optimizer.zero_grad()

        # Mixed precision forward (device-agnostic)
        with get_autocast_context(device_info, enabled=use_amp):
            logits, encoder_lengths = model(mels, mel_lengths, tokens, token_lengths)

            # RNN-T loss (compute on CPU for ROCm compatibility)
            loss = F_audio.rnnt_loss(
                logits.float().cpu(),
                tokens.int().cpu(),
                encoder_lengths.int().cpu(),
                token_lengths.int().cpu(),
                blank=blank_id,
                reduction='mean',
            ).to(device)

        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            nan_loss_count += 1
            # Debug: print constraint check
            enc_lens = encoder_lengths.tolist()
            tok_lens = token_lengths.tolist()
            violations = [(i, e, t) for i, (e, t) in enumerate(zip(enc_lens, tok_lens)) if e < t]
            print(f"\n[NaN Loss #{nan_loss_count}] batch={batch_idx}")
            print(f"  enc_lens={enc_lens}")
            print(f"  tok_lens={tok_lens}")
            if violations:
                print(f"  CONSTRAINT VIOLATED: {violations}")
            else:
                print(f"  Constraint OK, but loss=NaN. Check logits.")
                logits_stats = f"min={logits.min().item():.2e}, max={logits.max().item():.2e}"
                print(f"  logits: {logits_stats}, has_nan={torch.isnan(logits).any().item()}")

            if nan_loss_count + nan_grad_count > max_nan_skip:
                raise RuntimeError(f"Too many NaN (loss={nan_loss_count}, grad={nan_grad_count})")
            continue

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training'].get('grad_clip', 1.0),
        )

        # Check for NaN gradients
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            nan_grad_count += 1
            print(f"\n[NaN Grad #{nan_grad_count}] batch={batch_idx}, loss={loss.item():.2f}, grad_norm={grad_norm}")
            optimizer.zero_grad()
            scaler.update()
            if nan_loss_count + nan_grad_count > max_nan_skip:
                raise RuntimeError(f"Too many NaN (loss={nan_loss_count}, grad={nan_grad_count})")
            continue

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Update meters
        loss_meter.update(loss.item())
        grad_norm_meter.update(grad_norm.item())

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.2f}',
            'lr': f'{current_lr:.2e}',
            'grad': f'{grad_norm_meter.avg:.1f}',
            'nan': nan_loss_count + nan_grad_count,
        })

        # Log to wandb
        if wandb_run is not None and batch_idx % 100 == 0:
            wandb_run.log({
                'train/loss': loss.item(),
                'train/lr': current_lr,
                'train/grad_norm': grad_norm.item(),
                'train/step': scheduler.n_steps,
            })

    # Print epoch summary
    if rank == 0:
        nan_total = nan_loss_count + nan_grad_count
        print(f"\n[Epoch {epoch}] loss={loss_meter.avg:.2f} | batches={total_batches} | "
              f"nan={nan_total} (loss={nan_loss_count}, grad={nan_grad_count})")

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
            device, device_info, epoch, config, rank, wandb_run
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
