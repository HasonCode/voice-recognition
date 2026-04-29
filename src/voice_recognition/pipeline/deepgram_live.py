"""Live microphone transcription via Deepgram Listen v1 (cloud WebSocket).

Requires ``deepgram-sdk`` and ``DEEPGRAM_API_KEY``. Install::

    pip install deepgram-sdk
    # or: poetry install -E deepgram

Example (same env var the official SDK uses): ``export DEEPGRAM_API_KEY=...``

Docs: https://developers.deepgram.com/docs/live-streaming-audio
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from voice_recognition.audio import AudioCollector
from voice_recognition.audio.config import AudioConfig


def _float32_pcm_to_linear16_bytes(chunk: np.ndarray) -> bytes:
    x = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


class DeepgramStreamingPipeline:
    """Mic → linear16 PCM → Deepgram WebSocket → ``on_display`` (same hook as NeMo pipeline)."""

    stabilizer = None  # NeMo path uses stabilizer for stuck-caption reset; Deepgram does not.

    def __init__(
        self,
        *,
        audio_config: Optional[AudioConfig] = None,
        chunk_duration_sec: float = 0.25,
        on_display: Optional[Callable[[str], None]] = None,
        asr_enabled: Optional[Callable[[], bool]] = None,
        model: str = "nova-3",
        api_key: Optional[str] = None,
    ):
        self.audio_config = audio_config or AudioConfig()
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.on_display = on_display or (lambda _s: None)
        self._asr_enabled = asr_enabled
        self.model = model
        self._api_key = (api_key or os.environ.get("DEEPGRAM_API_KEY") or "").strip()
        self.audio_collector = AudioCollector(self.audio_config)
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def _asr_allowed(self) -> bool:
        return self._asr_enabled is None or self._asr_enabled()

    def run(self, device: Optional[int] = None) -> None:
        if not self._api_key:
            raise ValueError(
                "Deepgram requires an API key: set environment variable DEEPGRAM_API_KEY "
                "or pass api_key= to DeepgramStreamingPipeline."
            )
        try:
            from deepgram import DeepgramClient
            from deepgram.core.events import EventType
            from deepgram.listen.v1.types import ListenV1Results
        except ImportError as e:
            raise ImportError(
                "The deepgram-sdk package is required for --deepgram. "
                "Install with: pip install deepgram-sdk"
            ) from e

        self._stopped = False
        client = DeepgramClient(api_key=self._api_key)

        listen_thread: Optional[threading.Thread] = None
        last_keepalive = 0.0

        # Query params must use Deepgram's string booleans ("true"/"false"), not Python bool,
        # or the server returns HTTP 400 on the WebSocket handshake.
        with client.listen.v1.connect(
            model=self.model,
            encoding="linear16",
            sample_rate=self.audio_config.sample_rate,
            channels=self.audio_config.channels,
            interim_results="true",
            smart_format="true",
            language="en-US",
        ) as connection:

            def on_message(message: object) -> None:
                if self._stopped:
                    return
                if isinstance(message, ListenV1Results):
                    ch = message.channel
                    if ch and ch.alternatives:
                        tr = (ch.alternatives[0].transcript or "").strip()
                        if tr:
                            self.on_display(tr)

            def on_err(err: object) -> None:
                print(f"[Deepgram] error: {err!r}")

            connection.on(EventType.MESSAGE, on_message)
            connection.on(EventType.ERROR, on_err)

            def listen_worker() -> None:
                try:
                    connection.start_listening()
                except Exception as ex:
                    if not self._stopped:
                        print(f"[Deepgram] listen thread exited: {ex!r}")

            listen_thread = threading.Thread(target=listen_worker, daemon=True)
            listen_thread.start()
            time.sleep(0.2)

            try:
                for chunk in self.audio_collector.record_stream(
                    chunk_duration_sec=self.chunk_duration_sec,
                    device=device,
                ):
                    if self._stopped:
                        break
                    if self._asr_allowed():
                        connection.send_media(_float32_pcm_to_linear16_bytes(chunk))
                    else:
                        now = time.time()
                        if now - last_keepalive >= 3.0:
                            try:
                                if hasattr(connection, "send_keep_alive"):
                                    connection.send_keep_alive()
                                elif hasattr(connection, "send_keepalive"):
                                    connection.send_keepalive()
                            except Exception:
                                pass
                            last_keepalive = now
            except KeyboardInterrupt:
                raise
            finally:
                self._stopped = True
                try:
                    if hasattr(connection, "send_close_stream"):
                        connection.send_close_stream()
                except Exception:
                    pass

            if listen_thread is not None:
                listen_thread.join(timeout=8.0)
