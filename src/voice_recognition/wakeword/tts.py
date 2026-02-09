"""
Text-to-speech for reading responses aloud. Tries edge-tts (AI-style voice)
then falls back to pyttsx3 (offline).
"""

import subprocess
import tempfile
import threading
from pathlib import Path


def _speak_edge_tts(text: str, voice: str = "en-US-AriaNeural") -> bool:
    """Use Microsoft Edge TTS (needs internet). Returns True if successful."""
    try:
        import edge_tts
    except ImportError:
        return False
    if not text or not text.strip():
        return True
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name
    try:
        import asyncio
        async def run():
            communicate = edge_tts.Communicate(text.strip(), voice)
            await communicate.save(path)
        asyncio.run(run())
        # Play: try mpv, then ffplay, then leave file for user
        for cmd in (
            ["mpv", "--no-video", "--really-quiet", path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ):
            try:
                subprocess.run(cmd, check=True, timeout=120, capture_output=True)
                return True
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        return False
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def _speak_pyttsx3(text: str) -> bool:
    """Use pyttsx3 (offline, e.g. espeak on Linux). Returns True if successful."""
    try:
        import pyttsx3
    except ImportError:
        return False
    if not text or not text.strip():
        return True
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text.strip())
        engine.runAndWait()
        return True
    except Exception:
        return False


def speak(text: str, voice: str = "en-US-AriaNeural", use_thread: bool = True):
    """
    Speak the given text aloud. Tries edge-tts first, then pyttsx3.
    If use_thread is True (default), runs in a background thread so the caller
    is not blocked.
    """
    def _run():
        if _speak_edge_tts(text, voice=voice):
            return
        _speak_pyttsx3(text)

    if use_thread:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
    else:
        _run()
