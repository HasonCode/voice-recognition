"""Centralized audio and feature extraction configuration.

Encoding standards:
- Audio: mono 16 kHz
- Features: 80-bin log-Mel filterbanks
- STFT: 25 ms window / 10 ms hop, FFT 512
- Streaming: ring buffer, online CMVN (sliding mean/var over a few seconds)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    """Audio recording and encoding configuration."""

    # Recording
    sample_rate: int = 16_000
    channels: int = 1  # mono
    dtype: str = "float32"

    # STFT
    window_ms: float = 25.0
    hop_ms: float = 10.0
    fft_size: int = 512

    # Mel filterbanks
    n_mels: int = 80

    # Online CMVN (Cepstral Mean and Variance Normalization)
    # Sliding window duration in seconds for streaming normalization
    cmvn_window_sec: float = 3.0

    @property
    def frame_length(self) -> int:
        """STFT window length in samples."""
        return int(self.sample_rate * self.window_ms / 1000)

    @property
    def hop_length(self) -> int:
        """STFT hop length in samples."""
        return int(self.sample_rate * self.hop_ms / 1000)

    @property
    def frames_per_second(self) -> float:
        """Number of feature frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def cmvn_window_frames(self) -> int:
        """CMVN sliding window size in frames."""
        return int(self.cmvn_window_sec * self.frames_per_second)
