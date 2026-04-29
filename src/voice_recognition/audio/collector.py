"""Audio collection at mono 16 kHz for voice recognition."""

import queue
import threading
from typing import Iterator, Optional

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
        self._buffer: list[np.ndarray] = []
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
        *,
        coalesce_queued_chunks: bool = True,
        stream_latency: str | float = "low",
    ) -> Iterator[np.ndarray]:
        """Stream audio chunks continuously.

        Args:
            chunk_duration_sec: Duration of each yielded chunk in seconds.
            device: Input device index (None = default).
            coalesce_queued_chunks: If True (default), drain all samples currently
                waiting in the queue and yield them as one array. When inference
                is slower than realtime, this prevents an unbounded backlog so
                transcription tracks live speech instead of lagging by many
                seconds. The downstream ring buffer already keeps only the tail
                window of audio.
            stream_latency: PortAudio latency hint, e.g. ``\"low\"`` (default)
                or seconds as float. Lower values reduce capture delay; too low
                may cause dropouts on some hardware.

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
            latency=stream_latency,
        ):
            while True:
                first = q.get()
                if not coalesce_queued_chunks:
                    yield first
                    continue
                pending: list[np.ndarray] = [first]
                while True:
                    try:
                        pending.append(q.get_nowait())
                    except queue.Empty:
                        break
                if len(pending) == 1:
                    yield pending[0]
                else:
                    yield np.concatenate(pending)

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
