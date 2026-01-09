"""
FastConformer-Transducer Inference Script

Optimized for 80ms low-latency streaming ASR.

Features:
- Offline mode: Full-context transcription with beam search
- Streaming mode: Real-time transcription with ~80ms latency
- Long audio support: Constant memory via sliding window cache
- Latency measurement: Track per-chunk and total latency

Usage:
  Offline: python inference.py --config config/fastconformer_small.json --audio audio.wav
  Streaming: python inference.py --config config/fastconformer_small.json --audio audio.wav --streaming
  Low-latency: python inference.py --config config/fastconformer_small.json --audio audio.wav --streaming --chunk-ms 80
"""

import os
import sys
import argparse
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio

from model import FastConformerTransducer, create_model
from utils import load_config, get_text_transform, get_device_info


@dataclass
class LatencyStats:
    """Statistics for streaming latency analysis."""
    total_audio_ms: float = 0.0
    total_inference_ms: float = 0.0
    chunk_latencies_ms: List[float] = None

    def __post_init__(self):
        if self.chunk_latencies_ms is None:
            self.chunk_latencies_ms = []

    @property
    def rtf(self) -> float:
        """Real-time factor."""
        if self.total_audio_ms == 0:
            return 0.0
        return self.total_inference_ms / self.total_audio_ms

    @property
    def avg_chunk_latency_ms(self) -> float:
        """Average per-chunk latency."""
        if not self.chunk_latencies_ms:
            return 0.0
        return sum(self.chunk_latencies_ms) / len(self.chunk_latencies_ms)

    @property
    def max_chunk_latency_ms(self) -> float:
        """Maximum per-chunk latency."""
        if not self.chunk_latencies_ms:
            return 0.0
        return max(self.chunk_latencies_ms)


class OfflineTranscriber:
    """
    Offline (full-context) transcription.

    Best for: Pre-recorded audio where latency is not critical.
    """

    def __init__(
        self,
        model: FastConformerTransducer,
        text_transform,
        device: torch.device,
        sample_rate: int = 16000,
        n_mels: int = 80,
    ):
        self.model = model
        self.text_transform = text_transform
        self.device = device
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=sample_rate // 100,  # 10ms hop
        ).to(device)

        self.model.eval()

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform

    @torch.inference_mode()
    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 1,
    ) -> Tuple[str, LatencyStats]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            beam_size: Beam size for decoding (1 = greedy)

        Returns:
            Tuple of (transcription, latency_stats)
        """
        # Load audio
        waveform = self.load_audio(audio_path).to(self.device)
        audio_duration_ms = waveform.size(1) / self.sample_rate * 1000

        start_time = time.perf_counter()

        # Compute mel spectrogram
        mel = self.mel_transform(waveform).squeeze(0).transpose(0, 1)  # (time, n_mels)
        mel = mel.unsqueeze(0)  # (1, time, n_mels)
        mel_lengths = torch.tensor([mel.size(1)], device=self.device)

        # Encode (offline mode - no cache)
        encoder_out, encoder_lengths, _ = self.model.encode(
            mel, mel_lengths, cache=None, streaming=False
        )

        # Decode
        if beam_size == 1:
            decoded = self._greedy_decode(encoder_out)
        else:
            decoded = self._beam_decode(encoder_out, beam_size)

        # Convert to text
        transcription = self.text_transform.int_to_text(decoded[0])

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        stats = LatencyStats(
            total_audio_ms=audio_duration_ms,
            total_inference_ms=inference_time_ms,
        )

        return transcription, stats

    def _greedy_decode(
        self,
        encoder_out: torch.Tensor,
        max_symbols_per_step: int = 10,
    ) -> List[List[int]]:
        """Greedy decoding."""
        batch_size, T, _ = encoder_out.size()
        device = encoder_out.device
        blank_id = self.model.blank_id

        decoded = [[] for _ in range(batch_size)]

        for b in range(batch_size):
            # Start with blank
            y = torch.zeros(1, 1, dtype=torch.long, device=device)
            predictor_out = self.model.predict(y)

            for t in range(T):
                enc_t = encoder_out[b:b+1, t:t+1, :]

                for _ in range(max_symbols_per_step):
                    logits = self.model.joint_step(enc_t, predictor_out)
                    pred = logits.argmax(dim=-1).item()

                    if pred == blank_id:
                        break

                    decoded[b].append(pred)

                    # Update predictor
                    y = torch.tensor([[pred]], dtype=torch.long, device=device)
                    predictor_out = self.model.predict(y)

        return decoded

    def _beam_decode(
        self,
        encoder_out: torch.Tensor,
        beam_size: int = 5,
        max_symbols_per_step: int = 10,
    ) -> List[List[int]]:
        """Beam search decoding."""
        batch_size, T, _ = encoder_out.size()
        device = encoder_out.device
        blank_id = self.model.blank_id

        all_decoded = []

        for b in range(batch_size):
            # Beam: (score, tokens, predictor_out)
            beams = [(0.0, [], None)]

            for t in range(T):
                enc_t = encoder_out[b:b+1, t:t+1, :]
                new_beams = []

                for score, tokens, pred_state in beams:
                    # Get predictor output
                    if pred_state is None:
                        y = torch.zeros(1, 1, dtype=torch.long, device=device)
                        predictor_out = self.model.predict(y)
                    else:
                        predictor_out = pred_state

                    # Single symbol expansion per frame
                    logits = self.model.joint_step(enc_t, predictor_out)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                    # Blank continuation
                    blank_score = score + log_probs[blank_id].item()
                    new_beams.append((blank_score, tokens.copy(), predictor_out))

                    # Non-blank expansions
                    topk_probs, topk_indices = torch.topk(log_probs, min(beam_size, log_probs.size(0)))
                    for prob, idx in zip(topk_probs, topk_indices):
                        if idx.item() == blank_id:
                            continue

                        new_tokens = tokens + [idx.item()]
                        new_score = score + prob.item()

                        y = torch.tensor([[idx.item()]], dtype=torch.long, device=device)
                        new_pred_out = self.model.predict(y)

                        new_beams.append((new_score, new_tokens, new_pred_out))

                # Keep top beams
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_size]

            # Get best beam
            best_beam = max(beams, key=lambda x: x[0])
            all_decoded.append(best_beam[1])

        return all_decoded


class StreamingTranscriber:
    """
    Low-latency streaming transcription.

    Optimized for ~80ms latency with proper cache management.
    Supports unlimited audio length with constant memory.

    Latency breakdown:
    - 80ms: 8 mel frames (8x subsampling) = 1 encoder frame
    - Processing: ~5-10ms on modern GPU
    - Total: ~85-95ms end-to-end
    """

    def __init__(
        self,
        model: FastConformerTransducer,
        text_transform,
        device: torch.device,
        sample_rate: int = 16000,
        n_mels: int = 80,
        chunk_size_ms: int = 80,  # Target 80ms latency
    ):
        self.model = model
        self.text_transform = text_transform
        self.device = device
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Chunk settings
        # 80ms = 1280 samples at 16kHz
        self.chunk_size_ms = chunk_size_ms
        self.chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)

        # Mel spectrogram transform
        # hop_length = 160 = 10ms, so 80ms = 8 mel frames
        self.hop_length = sample_rate // 100  # 10ms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=self.hop_length,
            n_fft=512,
            win_length=400,  # 25ms window
        ).to(device)

        self.model.eval()
        self.reset()

    def reset(self):
        """Reset streaming state for new audio."""
        self.encoder_cache = None
        self.audio_buffer = torch.tensor([], device=self.device)
        self.mel_buffer = torch.tensor([], device=self.device)

        # Predictor state
        self.last_token = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.predictor_out = None

        # Decoded tokens
        self.decoded_tokens = []

        # Stats
        self.total_audio_samples = 0
        self.chunk_latencies = []

    @torch.inference_mode()
    def process_chunk(self, audio_chunk: torch.Tensor) -> Tuple[str, float]:
        """
        Process an audio chunk and return incremental transcription.

        Args:
            audio_chunk: (samples,) raw audio samples

        Returns:
            Tuple of (incremental_text, chunk_latency_ms)
        """
        chunk_start = time.perf_counter()

        # Add to buffer
        audio_chunk = audio_chunk.to(self.device)
        self.audio_buffer = torch.cat([self.audio_buffer, audio_chunk])
        self.total_audio_samples += audio_chunk.size(0)

        # Check if we have enough audio for a full chunk
        if self.audio_buffer.size(0) < self.chunk_size_samples:
            return "", 0.0

        # Extract exactly one chunk
        chunk = self.audio_buffer[:self.chunk_size_samples].unsqueeze(0)
        self.audio_buffer = self.audio_buffer[self.chunk_size_samples:]

        # Compute mel spectrogram for this chunk
        mel = self.mel_transform(chunk).squeeze(0).transpose(0, 1)  # (time, n_mels)

        # Skip if no frames produced
        if mel.size(0) == 0:
            return "", 0.0

        mel = mel.unsqueeze(0)  # (1, time, n_mels)

        # Encode with cache
        encoder_out, _, new_cache = self.model.encode(
            mel,
            lengths=None,
            cache=self.encoder_cache,
            streaming=True,
        )
        self.encoder_cache = new_cache

        # Decode new frames
        new_tokens = self._decode_incremental(encoder_out)
        self.decoded_tokens.extend(new_tokens)

        chunk_latency_ms = (time.perf_counter() - chunk_start) * 1000
        self.chunk_latencies.append(chunk_latency_ms)

        # Return incremental text
        if new_tokens:
            return self.text_transform.int_to_text(new_tokens), chunk_latency_ms
        return "", chunk_latency_ms

    def _decode_incremental(
        self,
        encoder_out: torch.Tensor,
        max_symbols_per_step: int = 10,
    ) -> List[int]:
        """Decode new encoder frames incrementally."""
        T = encoder_out.size(1)
        blank_id = self.model.blank_id
        new_tokens = []

        # Initialize predictor if needed
        if self.predictor_out is None:
            self.predictor_out = self.model.predict(self.last_token)

        for t in range(T):
            enc_t = encoder_out[:, t:t+1, :]

            for _ in range(max_symbols_per_step):
                logits = self.model.joint_step(enc_t, self.predictor_out)
                pred = logits.argmax(dim=-1).item()

                if pred == blank_id:
                    break

                new_tokens.append(pred)
                self.last_token = torch.tensor([[pred]], dtype=torch.long, device=self.device)
                self.predictor_out = self.model.predict(self.last_token)

        return new_tokens

    @torch.inference_mode()
    def finalize(self) -> str:
        """Process remaining audio and return final transcription."""
        # Process any remaining audio in buffer
        if self.audio_buffer.size(0) > 0:
            chunk = self.audio_buffer.unsqueeze(0)
            mel = self.mel_transform(chunk).squeeze(0).transpose(0, 1)

            if mel.size(0) > 0:
                mel = mel.unsqueeze(0)
                encoder_out, _, _ = self.model.encode(
                    mel, None, self.encoder_cache, streaming=True
                )
                new_tokens = self._decode_incremental(encoder_out)
                self.decoded_tokens.extend(new_tokens)

        return self.text_transform.int_to_text(self.decoded_tokens)

    def transcribe_file(
        self,
        audio_path: str,
        return_partial: bool = False,
        simulate_realtime: bool = False,
    ) -> Tuple[str, LatencyStats, Optional[List[Tuple[float, str]]]]:
        """
        Transcribe a file using streaming mode.

        Args:
            audio_path: Path to audio file
            return_partial: Return partial transcriptions with timestamps
            simulate_realtime: Sleep to simulate real-time audio arrival

        Returns:
            Tuple of (final_transcription, latency_stats, partial_results)
        """
        self.reset()

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)
        audio_duration_ms = waveform.size(0) / self.sample_rate * 1000

        partial_results = [] if return_partial else None
        total_start = time.perf_counter()

        # Process in chunks
        num_chunks = (waveform.size(0) + self.chunk_size_samples - 1) // self.chunk_size_samples
        current_time_ms = 0.0

        for i in range(num_chunks):
            start_idx = i * self.chunk_size_samples
            end_idx = min(start_idx + self.chunk_size_samples, waveform.size(0))
            chunk = waveform[start_idx:end_idx]

            # Simulate real-time if requested
            if simulate_realtime and i > 0:
                time.sleep(self.chunk_size_ms / 1000)

            incremental_text, chunk_latency = self.process_chunk(chunk)
            current_time_ms = end_idx / self.sample_rate * 1000

            if return_partial and incremental_text:
                partial_results.append((current_time_ms / 1000, incremental_text))

        # Finalize
        final_text = self.finalize()

        total_inference_ms = (time.perf_counter() - total_start) * 1000

        stats = LatencyStats(
            total_audio_ms=audio_duration_ms,
            total_inference_ms=total_inference_ms,
            chunk_latencies_ms=self.chunk_latencies.copy(),
        )

        return final_text, stats, partial_results


class RealtimeTranscriber:
    """
    Real-time transcriber for live audio input.

    Designed for microphone input or audio streaming.
    Prints transcription as it becomes available.
    """

    def __init__(
        self,
        model: FastConformerTransducer,
        text_transform,
        device: torch.device,
        sample_rate: int = 16000,
        n_mels: int = 80,
        chunk_size_ms: int = 80,
        callback=None,
    ):
        self.streamer = StreamingTranscriber(
            model, text_transform, device,
            sample_rate, n_mels, chunk_size_ms
        )
        self.callback = callback or (lambda text: print(text, end='', flush=True))
        self.is_running = False

    def start(self):
        """Start real-time transcription."""
        self.streamer.reset()
        self.is_running = True

    def stop(self) -> str:
        """Stop and return final transcription."""
        self.is_running = False
        return self.streamer.finalize()

    def feed_audio(self, audio_chunk: torch.Tensor):
        """Feed audio chunk and trigger callback with transcription."""
        if not self.is_running:
            return

        text, _ = self.streamer.process_chunk(audio_chunk)
        if text:
            self.callback(text)


def load_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[FastConformerTransducer, dict]:
    """Load model from config and checkpoint."""
    config = load_config(config_path)
    model = create_model(config)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        wer = checkpoint.get('wer', 'N/A')
        print(f"  Loaded model from epoch {epoch}, WER: {wer}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("  Using randomly initialized model")

    model = model.to(device)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(
        description='FastConformer-Transducer Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline transcription
  python inference.py --config config/fastconformer_small.json --audio audio.wav

  # Streaming with 80ms latency
  python inference.py --config config/fastconformer_small.json --audio audio.wav --streaming

  # Show partial results
  python inference.py --config config/fastconformer_small.json --audio audio.wav --streaming --show-partial

  # Beam search decoding
  python inference.py --config config/fastconformer_small.json --audio audio.wav --beam-size 5
        """
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--streaming', action='store_true', help='Use streaming mode')
    parser.add_argument('--beam-size', type=int, default=1, help='Beam size (offline only)')
    parser.add_argument('--chunk-ms', type=int, default=80, help='Chunk size in ms (streaming)')
    parser.add_argument('--show-partial', action='store_true', help='Show partial results')
    parser.add_argument('--simulate-realtime', action='store_true', help='Simulate real-time')
    args = parser.parse_args()

    # Get device (auto-detect CPU/CUDA/ROCm)
    device_info = get_device_info()
    device = torch.device(device_info['device'])

    # Load config
    config = load_config(args.config)

    print(f"\n{'='*60}")
    print(f"FastConformer-Transducer Inference")
    print(f"Device: {device_info['device_name']}")
    print(f"{'='*60}")

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = config.get('training', {}).get('checkpoint_dir', './checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # Load model
    model, config = load_model(args.config, checkpoint_path, device)

    # Text transform
    text_transform = get_text_transform()

    print(f"\nTranscribing: {args.audio}")
    print(f"Mode: {'Streaming' if args.streaming else 'Offline'}")
    if args.streaming:
        print(f"Chunk size: {args.chunk_ms}ms")
    print()

    if args.streaming:
        # Streaming transcription
        transcriber = StreamingTranscriber(
            model,
            text_transform,
            device,
            sample_rate=config.get('data', {}).get('sample_rate', 16000),
            n_mels=config.get('model', {}).get('d_input', 80),
            chunk_size_ms=args.chunk_ms,
        )

        transcription, stats, partial = transcriber.transcribe_file(
            args.audio,
            return_partial=args.show_partial,
            simulate_realtime=args.simulate_realtime,
        )

        if args.show_partial and partial:
            print("Partial Results:")
            for timestamp, text in partial:
                print(f"  [{timestamp:.2f}s] {text}")
            print()

    else:
        # Offline transcription
        transcriber = OfflineTranscriber(
            model,
            text_transform,
            device,
            sample_rate=config.get('data', {}).get('sample_rate', 16000),
            n_mels=config.get('model', {}).get('d_input', 80),
        )

        transcription, stats = transcriber.transcribe(args.audio, beam_size=args.beam_size)

    # Print results
    print(f"Transcription: {transcription}")
    print(f"\n{'='*60}")
    print("Latency Statistics:")
    print(f"  Audio duration: {stats.total_audio_ms:.1f}ms ({stats.total_audio_ms/1000:.2f}s)")
    print(f"  Inference time: {stats.total_inference_ms:.1f}ms")
    print(f"  RTF: {stats.rtf:.3f}x")

    if args.streaming and stats.chunk_latencies_ms:
        print(f"  Avg chunk latency: {stats.avg_chunk_latency_ms:.1f}ms")
        print(f"  Max chunk latency: {stats.max_chunk_latency_ms:.1f}ms")
        print(f"  Theoretical latency: {args.chunk_ms + stats.avg_chunk_latency_ms:.1f}ms")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
