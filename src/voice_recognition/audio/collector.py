"""Audio collection at mono 16 kHz for voice recognition."""

import queue
import threading
from typing import Iterator, List, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore

from voice_recognition.audio.config import AudioConfig


class AudioCollector:
    """Records audio as mono 16 kHz PCM in streaming or batch mode."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._buffer: List[np.ndarray] = []
        self._lock = threading.Lock()

    def record_chunk(
        self,
        duration_sec: float,
        device: Optional[int] = None,
    ) -> np.ndarray:
        """Record a single chunk of audio.

        Args:
            duration_sec: Recording duration in seconds.
            device: Input device index (None = default).

        Returns:
            Mono float32 array, shape (n_samples,), normalized [-1, 1].
        """
        if sd is None:
            raise ImportError("sounddevice is required for recording. pip install sounddevice")

        samples = int(duration_sec * self.config.sample_rate)
        rec = sd.rec(
            samples,
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
        return rec.squeeze()

    def record_stream(
        self,
        chunk_duration_sec: float = 0.1,
        device: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """Stream audio chunks continuously.

        Args:
            chunk_duration_sec: Duration of each yielded chunk in seconds.
            device: Input device index (None = default).

        Yields:
            Mono float32 chunks, shape (n_samples,).
        """
        if sd is None:
            raise ImportError("sounddevice is required for recording. pip install sounddevice")

        chunk_samples = int(chunk_duration_sec * self.config.sample_rate)
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata: np.ndarray, _frames: int, _time: object, _status: object) -> None:
            q.put(indata.copy().squeeze())

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            blocksize=chunk_samples,
            device=device,
            callback=callback,
        ):
            while True:
                yield q.get()

    def record_to_file(
        self,
        filepath: str,
        duration_sec: float,
        device: Optional[int] = None,
    ) -> None:
        """Record audio and save as mono 16 kHz WAV.

        Args:
            filepath: Output path (e.g. .wav).
            duration_sec: Recording duration in seconds.
            device: Input device index (None = default).
        """
        import scipy.io.wavfile as wavfile

        audio = self.record_chunk(duration_sec, device=device)
        wavfile.write(
            filepath,
            self.config.sample_rate,
            (audio * 32767).astype(np.int16),
        )
