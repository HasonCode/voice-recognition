"""End-to-end streaming loop: audio -> mel -> model -> decoder -> stabilizer -> display.

Glue that wires existing components with minimal deps. Model is a callable
(model_forward) so you can plug PyTorch / ONNX / TensorRT.

Streaming: 1.6 s rolling context, update every 250 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from voice_recognition.audio import AudioCollector, MelFeatureExtractor
from voice_recognition.postprocess import refine_bpe_caption
from voice_recognition.audio.config import AudioConfig
from voice_recognition.audio.features import RingBuffer


@dataclass
class StreamingConfig:
    """Streaming caption pipeline parameters."""

    context_sec: float = 1.6
    update_interval_sec: float = 0.25
    sample_rate: int = 16_000

    @property
    def context_samples(self) -> int:
        return int(self.context_sec * self.sample_rate)

    @property
    def chunk_samples(self) -> int:
        return int(self.update_interval_sec * self.sample_rate)


# Model forward: accepts mel (T, 80) or audio (samples,) depending on model_input
ModelForward = Callable[[np.ndarray], np.ndarray]
DisplayCallback = Callable[[str], None]
PostProcessCallback = Callable[[str], str]


class StreamingCaptionPipeline:
    """Runs the full pipeline in a loop: audio -> mel -> model -> decoder -> stabilizer -> display.

    Components are injected so you can use real or mock audio, and any
    model (PyTorch/ONNX/TensorRT) via a callable.

    Interface:
      pipeline = StreamingCaptionPipeline(
          config=StreamingConfig(),
          mel_extractor=MelFeatureExtractor(),
          model_forward=my_model_fn,
          decoder=CTCPrefixBeamSearch(...),
          stabilizer=CaptionStabilizer(),
          on_display=print,
      )
      pipeline.run()  # blocks; use stop() from another thread or pass audio_iterator with sentinel
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        audio_config: Optional[AudioConfig] = None,
        mel_extractor: Optional[MelFeatureExtractor] = None,
        model_forward: Optional[ModelForward] = None,
        model_input: str = "mel",
        decoder: Optional[object] = None,
        post_process: Optional[PostProcessCallback] = None,
        stabilizer: Optional[object] = None,
        on_display: Optional[DisplayCallback] = None,
        audio_collector: Optional[AudioCollector] = None,
    ):
        self.streaming_config = config or StreamingConfig()
        self.audio_config = audio_config or AudioConfig()
        self.mel_extractor = mel_extractor or MelFeatureExtractor(self.audio_config)
        self.model_forward = model_forward
        self.model_input = model_input if model_input in ("mel", "audio") else "mel"
        self.decoder = decoder
        self.post_process = post_process if post_process is not None else refine_bpe_caption
        self.stabilizer = stabilizer
        self.on_display = on_display or (lambda s: None)
        self.audio_collector = audio_collector or AudioCollector(self.audio_config)

        self._audio_ring: Optional[RingBuffer] = None
        self._stopped = False

    def stop(self) -> None:
        """Signal the run loop to exit (checked each iteration)."""
        self._stopped = True

    def _ensure_audio_ring(self) -> RingBuffer:
        if self._audio_ring is None:
            self._audio_ring = RingBuffer(
                size=self.streaming_config.context_samples + self.streaming_config.chunk_samples,
                dtype=np.float32,
            )
        return self._audio_ring

    def _process_context(self, audio: np.ndarray) -> Optional[str]:
        """Run mel/audio -> model -> decoder -> stabilizer on one context window. Returns display text."""
        if len(audio) < self.streaming_config.context_samples:
            return None
        window = audio[-self.streaming_config.context_samples :].astype(np.float32)
        if self.model_forward is None or self.decoder is None or self.stabilizer is None:
            return None
        if self.model_input == "audio":
            model_input_data = window
        else:
            mel = self.mel_extractor.extract(window)
            if mel.shape[0] == 0:
                return None
            model_input_data = mel
        log_probs = self.model_forward(model_input_data)
        if log_probs is None or log_probs.size == 0:
            return None
        # Handle (1, T, V) from some frameworks
        if log_probs.ndim == 3:
            log_probs = log_probs.squeeze(0)
        partial, _ = self.decoder.decode(log_probs)
        partial = self.post_process(partial)
        display_text = self.stabilizer.update(partial)
        return display_text

    def run(
        self,
        audio_iterator: Optional[Iterator[np.ndarray]] = None,
        device: Optional[int] = None,
    ) -> None:
        """Run the streaming loop until stopped or iterator exhausted.

        Args:
            audio_iterator: If provided, use this as the source of audio chunks
                (each chunk = chunk_samples). If None, use microphone via
                audio_collector.record_stream(update_interval_sec).
            device: Microphone device index when using live audio (ignored if
                audio_iterator is provided).
        """
        self._stopped = False
        ring = self._ensure_audio_ring()
        chunk_samples = self.streaming_config.chunk_samples
        interval_sec = self.streaming_config.update_interval_sec

        if audio_iterator is not None:
            for chunk in audio_iterator:
                if self._stopped:
                    break
                chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.size == 0:
                    continue
                ring.push(chunk)
                display_text = self._process_context(ring.get_all())
                if display_text is not None:
                    self.on_display(display_text)
        else:
            for chunk in self.audio_collector.record_stream(
                chunk_duration_sec=interval_sec,
                device=device,
            ):
                if self._stopped:
                    break
                ring.push(chunk)
                display_text = self._process_context(ring.get_all())
                if display_text is not None:
                    self.on_display(display_text)

    def run_for_n_updates(
        self,
        n: int,
        audio_iterator: Iterator[np.ndarray],
    ) -> list[str]:
        """Run for exactly n context updates; used for tests. Returns list of display strings."""
        self._stopped = False
        ring = self._ensure_audio_ring()
        chunk_samples = self.streaming_config.chunk_samples
        displays: list[str] = []
        for chunk in audio_iterator:
            if len(displays) >= n:
                break
            chunk = np.asarray(chunk, dtype=np.float32)
            ring.push(chunk)
            display_text = self._process_context(ring.get_all())
            if display_text is not None:
                displays.append(display_text)
                self.on_display(display_text)
        return displays
