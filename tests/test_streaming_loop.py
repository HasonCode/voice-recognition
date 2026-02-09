"""Unit tests and toy example for streaming caption pipeline."""

from __future__ import annotations

import unittest
from typing import List

import numpy as np

from voice_recognition.audio import MelFeatureExtractor
from voice_recognition.audio.config import AudioConfig
from voice_recognition.decoder import CTCPrefixBeamSearch
from voice_recognition.pipeline import StreamingCaptionPipeline, StreamingConfig
from voice_recognition.stabilizer import CaptionStabilizer


# Tiny vocab for tests
VOCAB = ["<blank>", "a", "b", "c", " "]
BLANK = 0


def _make_dummy_model(vocab_size: int = len(VOCAB), bias_blank: bool = True):
    """Returns a callable (mel -> log_probs) for testing."""

    def forward(mel: np.ndarray) -> np.ndarray:
        T = mel.shape[0]
        V = vocab_size
        log_probs = np.random.randn(T, V).astype(np.float32) * 0.1
        if bias_blank:
            log_probs[:, BLANK] += 1.0
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))
        return log_probs

    return forward


def _fake_audio_stream(
    chunk_samples: int,
    num_chunks: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """Yield fake audio chunks (for testing)."""
    rng = np.random.default_rng(seed)
    return [
        rng.random(chunk_samples, dtype=np.float32) * 0.1
        for _ in range(num_chunks)
    ]


class TestStreamingCaptionPipeline(unittest.TestCase):
    """Tests for StreamingCaptionPipeline."""

    def setUp(self) -> None:
        self.config = StreamingConfig(
            context_sec=0.5,
            update_interval_sec=0.25,
            sample_rate=16_000,
        )
        self.audio_config = AudioConfig(sample_rate=16_000)
        self.mel_extractor = MelFeatureExtractor(self.audio_config)
        self.decoder = CTCPrefixBeamSearch(VOCAB, blank_index=BLANK, beam_size=8)
        self.stabilizer = CaptionStabilizer(stable_n=2)

    def test_process_context_returns_none_without_model(self) -> None:
        """Without model/decoder/stabilizer, _process_context returns None."""
        pipeline = StreamingCaptionPipeline(
            config=self.config,
            audio_config=self.audio_config,
            mel_extractor=self.mel_extractor,
            model_forward=None,
            decoder=None,
            stabilizer=None,
        )
        chunk_samples = self.config.chunk_samples
        context_samples = self.config.context_samples
        ring = pipeline._ensure_audio_ring()
        for _ in range((context_samples // chunk_samples) + 1):
            ring.push(np.zeros(chunk_samples, dtype=np.float32))
        audio = ring.get_all()
        out = pipeline._process_context(audio)
        self.assertIsNone(out)

    def test_run_for_n_updates_produces_displays(self) -> None:
        """With dummy model, run_for_n_updates returns display strings."""
        pipeline = StreamingCaptionPipeline(
            config=self.config,
            audio_config=self.audio_config,
            mel_extractor=self.mel_extractor,
            model_forward=_make_dummy_model(),
            decoder=self.decoder,
            stabilizer=self.stabilizer,
        )
        chunks = _fake_audio_stream(
            self.config.chunk_samples,
            num_chunks=20,
        )
        displays = pipeline.run_for_n_updates(3, iter(chunks))
        self.assertIsInstance(displays, list)
        self.assertGreaterEqual(len(displays), 1)
        for d in displays:
            self.assertIsInstance(d, str)

    def test_run_for_n_updates_integration(self) -> None:
        """Full pipeline: audio -> mel -> model -> decoder -> stabilizer."""
        collected: List[str] = []

        pipeline = StreamingCaptionPipeline(
            config=self.config,
            audio_config=self.audio_config,
            mel_extractor=self.mel_extractor,
            model_forward=_make_dummy_model(),
            decoder=self.decoder,
            stabilizer=self.stabilizer,
            on_display=collected.append,
        )
        chunks = _fake_audio_stream(
            self.config.chunk_samples,
            num_chunks=15,
        )
        pipeline.run_for_n_updates(5, iter(chunks))
        self.assertGreaterEqual(len(collected), 1)
        self.assertTrue(all(isinstance(x, str) for x in collected))


def run_toy_example() -> None:
    """Toy: run pipeline for a few updates with dummy model and print display."""
    print("=== Toy example: streaming caption pipeline ===\n")
    config = StreamingConfig(context_sec=0.5, update_interval_sec=0.25)
    audio_config = AudioConfig(sample_rate=16_000)
    decoder = CTCPrefixBeamSearch(VOCAB, blank_index=BLANK, beam_size=8)
    stabilizer = CaptionStabilizer(stable_n=2)

    pipeline = StreamingCaptionPipeline(
        config=config,
        audio_config=audio_config,
        model_forward=_make_dummy_model(),
        decoder=decoder,
        stabilizer=stabilizer,
        on_display=lambda s: print("  display:", repr(s)),
    )
    chunks = _fake_audio_stream(config.chunk_samples, num_chunks=12)
    displays = pipeline.run_for_n_updates(4, iter(chunks))
    print(f"\nCollected {len(displays)} display updates.")
    print("Done.")


if __name__ == "__main__":
    run_toy_example()
    print("\n--- Running unit tests ---")
    unittest.main(argv=[""], exit=False, verbosity=2)
