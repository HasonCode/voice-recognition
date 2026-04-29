"""Wake-word / trigger phrase detection and response capture until pause."""

from voice_recognition.wakeword.hey_jetson import (
    DEFAULT_DONE_PHRASES,
    DEFAULT_SIMILARITY_THRESHOLD,
    HeyJetsonListener,
)
from voice_recognition.wakeword.tts import play_audio_file, speak

__all__ = [
    "DEFAULT_DONE_PHRASES",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "HeyJetsonListener",
    "play_audio_file",
    "speak",
]
