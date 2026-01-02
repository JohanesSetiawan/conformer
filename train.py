#!/usr/bin/env python3
"""
Conformer ASR Training Script

Usage:
  python train.py --config config/conformer_small.json

Supports:
  - NVIDIA CUDA GPUs
  - AMD ROCm GPUs (MI300X, etc.)
  - Multi-GPU training (auto-detected)
  - Mixed precision training
  - Memory-efficient training
  - Weights & Biases logging
"""

import os
import gc
import argparse
import random
import torchaudio
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from model import ConformerEncoder, LSTMDecoder
from utils import (
    load_config,
    get_device_info,
    setup_distributed,
    cleanup_distributed,
    get_sorted_indices_lazy,
    BatchSampler,
    MemoryEfficientBatchSampler,
    preprocess_example,
    TransformerLrScheduler,
    GreedyCharacterDecoder,
    AvgMeter,
    model_size,
    add_model_noise,
    load_checkpoint,
    save_checkpoint,
    get_text_transform,
    clear_memory,
)

# Only argparse for config selection
parser = argparse.ArgumentParser("conformer")
parser.add_argument('--config', type=str, required=True, help='Path to JSON config file')
args = parser.parse_args()

# Global wandb run (initialized later if enabled)
wandb_run = None


def init_wandb(config, device_info):
    """Initialize Weights & Biases if enabled in config."""
    global wandb_run

    if not config.get('wandb', {}).get('enabled', False):
        return None

    try:
        import wandb

        wandb_config = config.get('wandb', {})
        project_name = wandb_config.get('project', 'conformer-asr')
        run_name = wandb_config.get('run_name', config['model']['name'])

        wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config={
                'model': config['model'],
                'training': config['training'],
                'data': config['data'],
                'hardware': config.get('hardware', {}),
                'device': device_info['device_name'],
            },
            reinit=True,
        )
        print(f"Weights & Biases initialized: {wandb_run.url}")
        return wandb_run
    except ImportError:
        print("Warning: wandb not installed. Install with: pip install wandb")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize wandb: {e}")
        return None


def log_wandb(metrics, step=None):
    """Log metrics to wandb if enabled."""
    global wandb_run
    if wandb_run is not None:
        try:
            import wandb
            wandb.log(metrics, step=step)
        except Exception:
            pass


def finish_wandb():
    """Finish wandb run."""
    global wandb_run
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


def create_dataloaders(config, device_info, rank=0, world_size=1):
    """Create train and test dataloaders with memory-efficient options."""
    data_config = config['data']
    training_config = config['training']

    # Load datasets
    if not os.path.isdir(data_config['data_dir']):
        os.makedirs(data_config['data_dir'], exist_ok=True)

    print(f"Loading training data: {data_config['train_set']}...")
    train_data = torchaudio.datasets.LIBRISPEECH(
        root=data_config['data_dir'],
        url=data_config['train_set'],
        download=True
    )

    print(f"Loading test data: {data_config['test_set']}...")
    test_data = torchaudio.datasets.LIBRISPEECH(
        data_config['data_dir'],
        url=data_config['test_set'],
        download=True
    )

    # Determine batch sampler strategy
    if data_config.get('smart_batch', True):
        print('Using smart batching for efficient training...')
        # Use memory-efficient sorting
        sorted_train_inds = get_sorted_indices_lazy(train_data)
        sorted_test_inds = get_sorted_indices_lazy(test_data)

        train_batch_sampler = BatchSampler(sorted_train_inds, batch_size=training_config['batch_size'])
        test_batch_sampler = BatchSampler(sorted_test_inds, batch_size=training_config['batch_size'])

        train_loader = DataLoader(
            dataset=train_data,
            pin_memory=data_config.get('pin_memory', True) if device_info['device'] != 'cpu' else False,
            num_workers=data_config.get('num_workers', 2),
            batch_sampler=train_batch_sampler,
            collate_fn=lambda x: preprocess_example(x, 'train'),
            prefetch_factor=data_config.get('prefetch_factor', 2) if data_config.get('num_workers', 2) > 0 else None,
        )

        test_loader = DataLoader(
            dataset=test_data,
            pin_memory=data_config.get('pin_memory', True) if device_info['device'] != 'cpu' else False,
            num_workers=data_config.get('num_workers', 2),
            batch_sampler=test_batch_sampler,
            collate_fn=lambda x: preprocess_example(x, 'valid'),
            prefetch_factor=data_config.get('prefetch_factor', 2) if data_config.get('num_workers', 2) > 0 else None,
        )
    else:
        # Standard dataloader without smart batching
        train_loader = DataLoader(
            dataset=train_data,
            pin_memory=data_config.get('pin_memory', True) if device_info['device'] != 'cpu' else False,
            num_workers=data_config.get('num_workers', 2),
            batch_size=training_config['batch_size'],
            shuffle=True,
            collate_fn=lambda x: preprocess_example(x, 'train'),
            prefetch_factor=data_config.get('prefetch_factor', 2) if data_config.get('num_workers', 2) > 0 else None,
        )

        test_loader = DataLoader(
            dataset=test_data,
            pin_memory=data_config.get('pin_memory', True) if device_info['device'] != 'cpu' else False,
            num_workers=data_config.get('num_workers', 2),
            batch_size=training_config['batch_size'],
            shuffle=False,
            collate_fn=lambda x: preprocess_example(x, 'valid'),
            prefetch_factor=data_config.get('prefetch_factor', 2) if data_config.get('num_workers', 2) > 0 else None,
        )

    return train_loader, test_loader


def create_model(config, device):
    """Create encoder and decoder models."""
    model_config = config['model']

    encoder = ConformerEncoder(
        d_input=model_config['d_input'],
        d_model=model_config['d_encoder'],
        num_layers=model_config['encoder_layers'],
        conv_kernel_size=model_config['conv_kernel_size'],
        dropout=model_config['dropout'],
        feed_forward_residual_factor=model_config['feed_forward_residual_factor'],
        feed_forward_expansion_factor=model_config['feed_forward_expansion_factor'],
        num_heads=model_config['attention_heads'],
    )

    decoder = LSTMDecoder(
        d_encoder=model_config['d_encoder'],
        d_decoder=model_config['d_decoder'],
        num_layers=model_config['decoder_layers'],
        num_classes=model_config.get('num_classes', 29),
    )

    # Move to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, decoder


def train_epoch(encoder, decoder, char_decoder, optimizer, scheduler, criterion,
                grad_scaler, train_loader, config, device, device_info, epoch):
    """Run a single training epoch with memory optimizations."""
    from torchmetrics.text.wer import WordErrorRate

    training_config = config['training']
    hardware_config = config.get('hardware', {})

    wer_metric = WordErrorRate()
    error_rate = AvgMeter()
    avg_loss = AvgMeter()
    text_transform = get_text_transform()

    encoder.train()
    decoder.train()

    accumulate_iters = training_config.get('accumulate_iters', 1)
    gradient_clip_value = training_config.get('gradient_clip_value', 1.0)
    empty_cache_freq = hardware_config.get('empty_cache_freq', 10)
    use_amp = training_config.get('use_amp', True)

    optimizer.zero_grad()

    # Store last batch predictions for epoch summary
    last_predictions = []
    last_references = []

    # Progress bar for training
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True, dynamic_ncols=True)

    for i, batch in enumerate(pbar):
        scheduler.step()

        # Periodic memory cleanup
        if i % empty_cache_freq == 0:
            clear_memory()

        spectrograms, labels, input_lengths, label_lengths, references, mask = batch

        # Move to device
        spectrograms = spectrograms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        input_lengths = torch.tensor(input_lengths, device=device)
        label_lengths = torch.tensor(label_lengths, device=device)
        mask = mask.to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            outputs = encoder(spectrograms, mask)
            outputs = decoder(outputs)
            loss = criterion(
                F.log_softmax(outputs, dim=-1).transpose(0, 1),
                labels,
                input_lengths,
                label_lengths
            )
            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulate_iters

        # Backward pass
        grad_scaler.scale(scaled_loss).backward()

        # Gradient accumulation step
        if (i + 1) % accumulate_iters == 0:
            # Unscale gradients for clipping
            grad_scaler.unscale_(optimizer)

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                gradient_clip_value
            )

            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        # Track loss
        avg_loss.update(loss.detach().item())

        # Compute predictions and WER
        with torch.no_grad():
            inds = char_decoder(outputs.detach())
            predictions = [text_transform.int_to_text(sample) for sample in inds]
            batch_wer = wer_metric(predictions, references) * 100
            error_rate.update(batch_wer)

            # Store for epoch summary (keep last batch)
            last_predictions = predictions
            last_references = references

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{avg_loss.avg:.4f}',
            'wer': f'{error_rate.avg:.2f}%',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

        # Log to wandb periodically
        if (i + 1) % 50 == 0:
            log_wandb({
                'train/step_loss': loss.detach().item(),
                'train/step_wer': batch_wer,
                'train/learning_rate': scheduler.get_last_lr()[0],
            })

        # Free memory
        del spectrograms, labels, mask, outputs

    pbar.close()

    return error_rate.avg if error_rate.avg else 0.0, avg_loss.avg, last_predictions, last_references


def validate(encoder, decoder, char_decoder, criterion, test_loader, config, device, device_info, epoch):
    """Evaluate model on test dataset."""
    from torchmetrics.text.wer import WordErrorRate

    training_config = config['training']
    use_amp = training_config.get('use_amp', True)

    avg_loss = AvgMeter()
    error_rate = AvgMeter()
    wer_metric = WordErrorRate()
    text_transform = get_text_transform()

    encoder.eval()
    decoder.eval()

    # Store last batch predictions for epoch summary
    last_predictions = []
    last_references = []

    # Progress bar for validation
    pbar = tqdm(test_loader, desc=f"Validation Epoch {epoch}", leave=True, dynamic_ncols=True)

    with torch.no_grad():
        for i, batch in enumerate(pbar):
            spectrograms, labels, input_lengths, label_lengths, references, mask = batch

            # Move to device
            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            input_lengths = torch.tensor(input_lengths, device=device)
            label_lengths = torch.tensor(label_lengths, device=device)
            mask = mask.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs = encoder(spectrograms, mask)
                outputs = decoder(outputs)
                loss = criterion(
                    F.log_softmax(outputs, dim=-1).transpose(0, 1),
                    labels,
                    input_lengths,
                    label_lengths
                )

            avg_loss.update(loss.item())

            inds = char_decoder(outputs.detach())
            predictions = [text_transform.int_to_text(sample) for sample in inds]
            batch_wer = wer_metric(predictions, references) * 100
            error_rate.update(batch_wer)

            # Store for epoch summary (keep last batch)
            last_predictions = predictions
            last_references = references

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{avg_loss.avg:.4f}',
                'wer': f'{error_rate.avg:.2f}%'
            })

            # Free memory
            del spectrograms, labels, mask, outputs

    pbar.close()

    return error_rate.avg, avg_loss.avg, last_predictions, last_references


def save_best_wer_checkpoint(encoder, decoder, wer, epoch, checkpoint_path):
    """Save checkpoint with best WER (model weights only)."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'best_wer': wer,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, checkpoint_path)


def main_worker(rank, world_size, config, device_info):
    """Main training worker (single or distributed)."""
    training_config = config['training']
    model_config = config['model']
    checkpoint_config = config['checkpoint']
    hardware_config = config.get('hardware', {})

    # Setup device
    if device_info['use_distributed']:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        if device_info['device'] == 'cuda':
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')

    is_main_process = rank == 0

    # Initialize wandb (only on main process)
    if is_main_process:
        init_wandb(config, device_info)

    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Conformer: {model_config['name']}")
        print(f"Device: {device_info['device_name']}")
        print(f"Mixed Precision: {training_config.get('use_amp', True)}")
        print(f"{'='*60}\n")

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(config, device_info, rank, world_size)

    # Create models
    encoder, decoder = create_model(config, device)

    # Print model info
    if is_main_process:
        model_size(encoder, 'Encoder')
        model_size(decoder, 'Decoder')

    # Wrap with DDP if distributed
    if device_info['use_distributed']:
        encoder = DDP(encoder, device_ids=[rank])
        decoder = DDP(decoder, device_ids=[rank])

    # Create other components
    char_decoder = GreedyCharacterDecoder().eval().to(device)
    criterion = nn.CTCLoss(blank=28, zero_infinity=True).to(device)

    # Optimizer with proper eps for mixed precision
    use_amp = training_config.get('use_amp', True)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-05 if use_amp else 1e-09,
        weight_decay=training_config.get('weight_decay', 1e-6)
    )

    scheduler = TransformerLrScheduler(
        optimizer,
        model_config['d_encoder'],
        training_config['warmup_steps']
    )

    # Mixed precision scaler
    grad_scaler = GradScaler(enabled=use_amp)

    # Load checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    best_wer = float('inf')

    if checkpoint_config.get('load_checkpoint', False):
        checkpoint_path = checkpoint_config['checkpoint_path']
        if os.path.exists(checkpoint_path):
            if is_main_process:
                print(f'Loading checkpoint from {checkpoint_path}...')

            # Handle DDP wrapped models
            enc_to_load = encoder.module if device_info['use_distributed'] else encoder
            dec_to_load = decoder.module if device_info['use_distributed'] else decoder

            start_epoch, best_loss = load_checkpoint(
                enc_to_load, dec_to_load, optimizer, scheduler, checkpoint_path
            )
            if is_main_process:
                print(f'Resumed training from epoch {start_epoch}')
        else:
            if is_main_process:
                print(f'No checkpoint found at {checkpoint_path}, starting fresh')

    # Ensure checkpoint directory exists
    checkpoint_dir = checkpoint_config.get('save_dir', 'checkpoints')
    if is_main_process and checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Define checkpoint paths
    resume_checkpoint_path = checkpoint_config.get('checkpoint_path', os.path.join(checkpoint_dir, 'checkpoint_latest.pt'))
    best_wer_checkpoint_path = checkpoint_config.get('best_wer_path', os.path.join(checkpoint_dir, 'best_wer.pt'))

    # Clear memory before training
    clear_memory()

    # Training loop
    for epoch in range(start_epoch, training_config['epochs']):
        current_epoch = epoch + 1

        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {current_epoch}/{training_config['epochs']}")
            print(f"{'='*60}")

        clear_memory()

        # Add variational noise for regularization
        noise_std = training_config.get('variational_noise_std', 0.0001)
        if noise_std > 0:
            enc_model = encoder.module if device_info['use_distributed'] else encoder
            dec_model = decoder.module if device_info['use_distributed'] else decoder
            add_model_noise(enc_model, std=noise_std, device=device)
            add_model_noise(dec_model, std=noise_std, device=device)

        # Train
        train_wer, train_loss, train_preds, train_refs = train_epoch(
            encoder, decoder, char_decoder, optimizer, scheduler,
            criterion, grad_scaler, train_loader, config, device, device_info, current_epoch
        )

        # Validate
        valid_wer, valid_loss, valid_preds, valid_refs = validate(
            encoder, decoder, char_decoder, criterion, test_loader,
            config, device, device_info, current_epoch
        )

        if is_main_process:
            # Select random samples for display (max 2)
            num_samples = min(2, len(valid_preds))
            if num_samples > 0:
                indices = random.sample(range(len(valid_preds)), num_samples)
                sample_preds = [valid_preds[i] for i in indices]
                sample_refs = [valid_refs[i] for i in indices]
            else:
                sample_preds = []
                sample_refs = []

            # Print epoch summary
            print(f"\nEpoch {current_epoch} Summary:")
            print(f"  Train WER: {train_wer:.2f}%, Train Loss: {train_loss:.4f}")
            print(f"  Valid WER: {valid_wer:.2f}%, Valid Loss: {valid_loss:.4f}")

            # Log to wandb
            log_wandb({
                'epoch': current_epoch,
                'train/epoch_wer': train_wer,
                'train/epoch_loss': train_loss,
                'valid/epoch_wer': valid_wer,
                'valid/epoch_loss': valid_loss,
                'learning_rate': scheduler.get_last_lr()[0],
            }, step=current_epoch)

            # Get models for saving
            enc_to_save = encoder.module if device_info['use_distributed'] else encoder
            dec_to_save = decoder.module if device_info['use_distributed'] else decoder

            # Save checkpoint if validation loss improved
            if valid_loss <= best_loss:
                print(f"  Validation loss improved ({best_loss:.4f} -> {valid_loss:.4f}), saving checkpoint...")
                best_loss = valid_loss

                save_checkpoint(
                    enc_to_save, dec_to_save, optimizer, scheduler,
                    valid_loss, current_epoch, resume_checkpoint_path
                )

            # Save best WER checkpoint
            if valid_wer <= best_wer:
                print(f"  Best WER improved ({best_wer:.2f}% -> {valid_wer:.2f}%), saving best WER checkpoint...")
                best_wer = valid_wer

                save_best_wer_checkpoint(
                    enc_to_save, dec_to_save, valid_wer, current_epoch, best_wer_checkpoint_path
                )

                # Log best WER to wandb
                log_wandb({'best_wer': best_wer})

            # Print sample predictions
            if sample_preds:
                print(f"  Sample Predictions: {sample_preds}")
                print(f"  Original References: {sample_refs}")

    # Cleanup
    if device_info['use_distributed']:
        cleanup_distributed()

    if is_main_process:
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"Best WER: {best_wer:.2f}%")
        print(f"{'='*60}")

        # Finish wandb
        finish_wandb()


def main():
    """Main entry point."""
    # Load configuration
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print("Conformer ASR Training")
    print(f"Config: {args.config}")
    print(f"{'='*60}\n")

    # Detect devices
    device_info = get_device_info()

    # Check distributed setting from config
    hardware_config = config.get('hardware', {})
    distributed_setting = hardware_config.get('distributed', 'auto')

    if distributed_setting == 'auto':
        # Use auto-detected setting
        pass
    elif distributed_setting == 'single':
        device_info['use_distributed'] = False
    elif distributed_setting == 'multi':
        if device_info['num_devices'] > 1:
            device_info['use_distributed'] = True
        else:
            print("Warning: Multi-GPU requested but only 1 GPU available")
            device_info['use_distributed'] = False

    # Run training
    if device_info['use_distributed']:
        world_size = device_info['num_devices']
        print(f"Starting distributed training with {world_size} GPUs...")
        mp.spawn(
            main_worker,
            args=(world_size, config, device_info),
            nprocs=world_size,
            join=True
        )
    else:
        main_worker(0, 1, config, device_info)


if __name__ == '__main__':
    main()
