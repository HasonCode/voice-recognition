"""Feature extraction: 80-bin log-Mel, STFT, ring buffer, online CMVN."""

from typing import Optional

import numpy as np

from voice_recognition.audio.config import AudioConfig


class RingBuffer:
    """Fixed-size ring buffer for continuous streaming audio/features."""

    def __init__(self, size: int, dtype: type = np.float32):
        self.size = size
        self.dtype = dtype
        self._data = np.zeros(size, dtype=dtype)
        self._write_idx = 0
        self._count = 0

    def push(self, chunk: np.ndarray) -> None:
        """Append a chunk; older data is overwritten."""
        n = len(chunk)
        if n >= self.size:
            self._data[:] = chunk[-self.size :].astype(self.dtype)
            self._write_idx = 0
            self._count = self.size
            return
        start = self._write_idx
        end = start + n
        if end <= self.size:
            self._data[start:end] = chunk.astype(self.dtype)
        else:
            head = self.size - start
            self._data[start:] = chunk[:head].astype(self.dtype)
            self._data[: end - self.size] = chunk[head:].astype(self.dtype)
        self._write_idx = end % self.size
        self._count = min(self._count + n, self.size)

    def get_all(self) -> np.ndarray:
        """Return all buffered data in chronological order."""
        if self._count == 0:
            return np.array([], dtype=self.dtype)
        if self._count < self.size:
            return self._data[: self._count].copy()
        return np.roll(self._data, -self._write_idx).copy()

    def clear(self) -> None:
        """Reset buffer."""
        self._write_idx = 0
        self._count = 0


def _mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: float,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Build Mel filterbank matrix."""
    if fmax is None:
        fmax = sample_rate / 2
    mel_points = np.linspace(
        _hz_to_mel(fmin),
        _hz_to_mel(fmax),
        n_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft)

    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        filters[i, left:center] = (np.arange(left, center) - left) / (center - left)
        filters[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


def _hz_to_mel(hz: float) -> float:
    return 2595 * np.log10(1 + hz / 700)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mel / 2595) - 1)


class MelFeatureExtractor:
    """Extract 80-bin log-Mel features with online CMVN for streaming."""

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._mel_filters = _mel_filterbank(
            self.config.n_mels,
            self.config.fft_size,
            float(self.config.sample_rate),
        )
        self._ring: Optional[RingBuffer] = None

    def stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT power spectrum.

        STFT: 25 ms window / 10 ms hop, FFT 512.
        """
        from scipy.signal import stft

        _, _, Zxx = stft(
            audio,
            fs=self.config.sample_rate,
            nperseg=self.config.frame_length,
            noverlap=self.config.frame_length - self.config.hop_length,
            nfft=self.config.fft_size,
            window="hann",
        )
        return (np.abs(Zxx) ** 2).T  # (n_frames, n_bins)

    def power_to_mel(self, power_spec: np.ndarray) -> np.ndarray:
        """Convert power spectrum to 80-bin log-Mel filterbanks."""
        mel = np.dot(power_spec, self._mel_filters.T)
        mel = np.maximum(mel, 1e-10)
        return np.log(mel).astype(np.float32)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-Mel features from raw audio (batch)."""
        power = self.stft(audio)
        return self.power_to_mel(power)

    def apply_online_cmvn(
        self,
        features: np.ndarray,
        frame_buffer: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply online CMVN: sliding mean/variance over a few seconds.

        Args:
            features: (n_frames, n_mels)
            frame_buffer: Optional buffer of recent frames for sliding window.
                          If None, uses only features for normalization.

        Returns:
            Normalized features.
        """
        window = self.config.cmvn_window_frames
        if frame_buffer is not None and len(frame_buffer) > 0:
            combined = np.vstack([frame_buffer, features])
        else:
            combined = features

        n = combined.shape[0]
        if n < 2:
            return features.copy()

        out = np.empty_like(features, dtype=np.float32)
        for i in range(features.shape[0]):
            idx = (frame_buffer.shape[0] if frame_buffer is not None else 0) + i
            start = max(0, idx - window + 1)
            windowed = combined[start : idx + 1]
            mean = windowed.mean(axis=0)
            std = windowed.std(axis=0)
            std = np.maximum(std, 1e-5)
            out[i] = (features[i] - mean) / std
        return out

    def extract_streaming(
        self,
        audio_chunk: np.ndarray,
        audio_ring: Optional[RingBuffer] = None,
        frame_ring: Optional[RingBuffer] = None,
        apply_cmvn: bool = True,
    ) -> tuple[np.ndarray, RingBuffer, RingBuffer]:
        """Extract features from a streaming audio chunk with ring buffers.

        Uses an audio ring buffer for STFT context and a frame ring buffer
        for online CMVN (sliding mean/var over a few seconds).

        Args:
            audio_chunk: Raw audio samples (mono 16 kHz).
            audio_ring: Ring buffer for raw audio (creates if None).
            frame_ring: Ring buffer for feature frames for CMVN (creates if None).
            apply_cmvn: Whether to apply online CMVN.

        Returns:
            (features, updated_audio_ring, updated_frame_ring)
        """
        cmvn_samples = int(
            self.config.cmvn_window_sec * self.config.sample_rate
        )
        overlap = self.config.frame_length - self.config.hop_length

        if audio_ring is None:
            audio_ring = RingBuffer(
                size=cmvn_samples + overlap + 4096,  # extra headroom
                dtype=np.float32,
            )
        if frame_ring is None:
            n_frames = self.config.cmvn_window_frames + 10
            frame_ring = RingBuffer(
                size=n_frames * self.config.n_mels,
                dtype=np.float32,
            )

        audio_ring.push(audio_chunk)
        audio = audio_ring.get_all()
        if len(audio) < self.config.frame_length:
            return (
                np.zeros((0, self.config.n_mels), dtype=np.float32),
                audio_ring,
                frame_ring,
            )

        features = self.extract(audio)

        if apply_cmvn and len(features) > 0:
            prev_frames = frame_ring.get_all()
            prev_frames = prev_frames.reshape(-1, self.config.n_mels)
            features = self.apply_online_cmvn(features, prev_frames)
            for f in features:
                frame_ring.push(f)

        return features, audio_ring, frame_ring
