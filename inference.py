"""
FastConformer-Transducer Inference Script

Supports both offline and streaming inference modes.

Usage:
  Offline: python inference.py --config config/fastconformer_small.json --audio audio.wav
  Streaming: python inference.py --config config/fastconformer_small.json --audio audio.wav --streaming
"""

import os
import argparse
import time
import torch
import torchaudio
import torch.nn.functional as F
from typing import List, Optional, Tuple

from model import FastConformerTransducer, create_model
from utils import load_config, get_text_transform, get_device_info


class OfflineTranscriber:
    """Offline (full-context) transcription."""

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

    def transcribe(self, audio_path: str, beam_size: int = 1) -> Tuple[str, float]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to audio file
            beam_size: Beam size for decoding (1 = greedy)

        Returns:
            Tuple of (transcription, RTF)
        """
        # Load audio
        waveform = self.load_audio(audio_path).to(self.device)
        audio_duration = waveform.size(1) / self.sample_rate

        start_time = time.time()

        with torch.no_grad():
            # Compute mel spectrogram
            mel = self.mel_transform(waveform).squeeze(0).transpose(0, 1)  # (time, n_mels)
            mel = mel.unsqueeze(0)  # (1, time, n_mels)
            mel_lengths = torch.tensor([mel.size(1)], device=self.device)

            # Encode
            encoder_out, encoder_lengths, _ = self.model.encode(
                mel, mel_lengths, mode="offline"
            )

            # Decode
            if beam_size == 1:
                decoded = self._greedy_decode(encoder_out)
            else:
                decoded = self._beam_decode(encoder_out, beam_size)

            # Convert to text
            transcription = self.text_transform.int_to_text(decoded[0])

        inference_time = time.time() - start_time
        rtf = inference_time / audio_duration

        return transcription, rtf

    def _greedy_decode(
        self,
        encoder_out: torch.Tensor,
        max_symbols: int = 100,
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

                for _ in range(max_symbols):
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
        max_symbols: int = 100,
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

                    # Expand beam
                    for _ in range(max_symbols):
                        logits = self.model.joint_step(enc_t, predictor_out)
                        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                        # Add blank continuation
                        blank_score = score + log_probs[blank_id].item()
                        new_beams.append((blank_score, tokens.copy(), predictor_out))

                        # Add non-blank continuations
                        topk = torch.topk(log_probs, min(beam_size, log_probs.size(0)))
                        for prob, idx in zip(topk.values, topk.indices):
                            if idx.item() == blank_id:
                                continue

                            new_tokens = tokens + [idx.item()]
                            new_score = score + prob.item()

                            # Update predictor
                            y = torch.tensor([[idx.item()]], dtype=torch.long, device=device)
                            new_pred_out = self.model.predict(y)

                            new_beams.append((new_score, new_tokens, new_pred_out))

                        # Only process blank once per frame
                        break

                # Keep top beams
                new_beams.sort(key=lambda x: x[0], reverse=True)
                beams = new_beams[:beam_size]

            # Get best beam
            best_beam = max(beams, key=lambda x: x[0])
            all_decoded.append(best_beam[1])

        return all_decoded


class StreamingTranscriber:
    """
    Streaming transcription with chunked processing.

    Processes audio in chunks with configurable lookahead for
    low-latency real-time transcription.
    """

    def __init__(
        self,
        model: FastConformerTransducer,
        text_transform,
        device: torch.device,
        sample_rate: int = 16000,
        n_mels: int = 80,
        chunk_size_ms: int = 160,  # Process 160ms chunks
        lookahead_ms: int = 80,    # 80ms lookahead
    ):
        self.model = model
        self.text_transform = text_transform
        self.device = device
        self.sample_rate = sample_rate

        # Chunk settings
        self.chunk_size_samples = int(sample_rate * chunk_size_ms / 1000)
        self.lookahead_samples = int(sample_rate * lookahead_ms / 1000)

        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=sample_rate // 100,  # 10ms hop
        ).to(device)

        self.model.eval()
        self.reset()

    def reset(self):
        """Reset streaming state."""
        self.encoder_cache = None
        self.audio_buffer = torch.tensor([], device=self.device)
        self.last_token = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.predictor_out = None
        self.decoded_tokens = []

    def process_chunk(self, audio_chunk: torch.Tensor) -> str:
        """
        Process an audio chunk and return incremental transcription.

        Args:
            audio_chunk: (samples,) raw audio samples

        Returns:
            New transcription (incremental)
        """
        # Add to buffer
        self.audio_buffer = torch.cat([self.audio_buffer, audio_chunk.to(self.device)])

        # Check if we have enough audio
        if self.audio_buffer.size(0) < self.chunk_size_samples:
            return ""

        # Extract chunk with lookahead
        chunk_end = min(
            self.chunk_size_samples + self.lookahead_samples,
            self.audio_buffer.size(0)
        )
        chunk = self.audio_buffer[:chunk_end].unsqueeze(0)

        # Compute mel spectrogram
        with torch.no_grad():
            mel = self.mel_transform(chunk).squeeze(0).transpose(0, 1)
            mel = mel.unsqueeze(0)
            mel_lengths = torch.tensor([mel.size(1)], device=self.device)

            # Encode with cache
            encoder_out, encoder_lengths, new_cache = self.model.encode(
                mel, mel_lengths, self.encoder_cache, mode="streaming"
            )
            self.encoder_cache = new_cache

            # Decode new frames
            new_tokens = self._decode_incremental(encoder_out)
            self.decoded_tokens.extend(new_tokens)

        # Remove processed audio (keep lookahead)
        self.audio_buffer = self.audio_buffer[self.chunk_size_samples:]

        # Return new text
        if new_tokens:
            return self.text_transform.int_to_text(new_tokens)
        return ""

    def _decode_incremental(
        self,
        encoder_out: torch.Tensor,
        max_symbols: int = 10,
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

            for _ in range(max_symbols):
                logits = self.model.joint_step(enc_t, self.predictor_out)
                pred = logits.argmax(dim=-1).item()

                if pred == blank_id:
                    break

                new_tokens.append(pred)
                self.last_token = torch.tensor([[pred]], dtype=torch.long, device=self.device)
                self.predictor_out = self.model.predict(self.last_token)

        return new_tokens

    def finalize(self) -> str:
        """Process any remaining audio and return final transcription."""
        # Process remaining buffer
        if self.audio_buffer.size(0) > 0:
            chunk = self.audio_buffer.unsqueeze(0)

            with torch.no_grad():
                mel = self.mel_transform(chunk).squeeze(0).transpose(0, 1)
                mel = mel.unsqueeze(0)
                mel_lengths = torch.tensor([mel.size(1)], device=self.device)

                encoder_out, _, _ = self.model.encode(
                    mel, mel_lengths, self.encoder_cache, mode="streaming"
                )

                new_tokens = self._decode_incremental(encoder_out)
                self.decoded_tokens.extend(new_tokens)

        return self.text_transform.int_to_text(self.decoded_tokens)

    def transcribe_file(
        self,
        audio_path: str,
        return_partial: bool = False,
    ) -> Tuple[str, float, Optional[List[Tuple[float, str]]]]:
        """
        Transcribe a file using streaming mode.

        Args:
            audio_path: Path to audio file
            return_partial: Whether to return partial transcriptions with timestamps

        Returns:
            Tuple of (final_transcription, RTF, partial_results)
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
        audio_duration = waveform.size(0) / self.sample_rate

        partial_results = [] if return_partial else None
        start_time = time.time()

        # Process in chunks
        chunk_size = self.chunk_size_samples
        num_chunks = (waveform.size(0) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, waveform.size(0))
            chunk = waveform[start:end]

            incremental_text = self.process_chunk(chunk)

            if return_partial and incremental_text:
                timestamp = (start + end) / 2 / self.sample_rate
                partial_results.append((timestamp, incremental_text))

        # Finalize
        final_text = self.finalize()

        inference_time = time.time() - start_time
        rtf = inference_time / audio_duration

        return final_text, rtf, partial_results


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """Load model from config and checkpoint."""
    config = load_config(config_path)
    model = create_model(config)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model from epoch {checkpoint['epoch']}, WER: {checkpoint.get('wer', 'N/A')}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description='FastConformer-Transducer Inference')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: from config)')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--streaming', action='store_true', help='Use streaming mode')
    parser.add_argument('--beam-size', type=int, default=1, help='Beam size for decoding')
    parser.add_argument('--chunk-size', type=int, default=160, help='Chunk size in ms (streaming)')
    parser.add_argument('--show-partial', action='store_true', help='Show partial results (streaming)')
    args = parser.parse_args()

    # Get device
    device_info = get_device_info()
    device = torch.device(device_info['device'])

    # Load config
    config = load_config(args.config)

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_dir = config['training'].get('checkpoint_dir', './checkpoints')
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

    # Load model
    model, config = load_model(args.config, checkpoint_path, device)

    # Text transform
    text_transform = get_text_transform()

    print(f"\nTranscribing: {args.audio}")
    print(f"Mode: {'Streaming' if args.streaming else 'Offline'}")
    print("-" * 50)

    if args.streaming:
        # Streaming transcription
        transcriber = StreamingTranscriber(
            model,
            text_transform,
            device,
            sample_rate=config['data'].get('sample_rate', 16000),
            n_mels=config['model'].get('d_input', 80),
            chunk_size_ms=args.chunk_size,
        )

        transcription, rtf, partial = transcriber.transcribe_file(
            args.audio,
            return_partial=args.show_partial,
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
            sample_rate=config['data'].get('sample_rate', 16000),
            n_mels=config['model'].get('d_input', 80),
        )

        transcription, rtf = transcriber.transcribe(args.audio, beam_size=args.beam_size)

    print(f"Transcription: {transcription}")
    print(f"RTF: {rtf:.3f}x")


if __name__ == '__main__':
    main()
