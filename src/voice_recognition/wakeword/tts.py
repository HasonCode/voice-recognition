"""
Text-to-speech for reading responses aloud. Tries edge-tts (AI-style voice)
then falls back to pyttsx3 (offline).
"""

import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import hashlib
from pathlib import Path

# pyttsx3 + espeak use ctypes callbacks after runAndWait(); a per-call engine can be
# collected first and triggers ReferenceError in the driver. Reuse one engine.
_pyttsx3_engine = None
_pyttsx3_lock = threading.Lock()
DEFAULT_APPLIO_RVC_CMD = (
    "python -m rvc_python cli --input {input_wav} --output {output_wav} "
    "--model {model_path} --index {index_path} --device cpu:0"
)
_cache_root = Path(tempfile.gettempdir()) / "voice_tts_cache"
_cache_lock = threading.Lock()


def _render_edge_tts_to_file(
    text: str,
    output_path: str,
    voice: str = "en-US-AriaNeural",
    speed: float = 1.0,
) -> bool:
    """Render Microsoft Edge TTS to an audio file. Returns True if successful."""
    try:
        import edge_tts
    except ImportError:
        return False
    if not text or not text.strip():
        return True
    try:
        import asyncio

        async def run():
            rate_pct = int(round((max(0.5, speed) - 1.0) * 100))
            rate = f"{rate_pct:+d}%"
            await edge_tts.Communicate(text.strip(), voice, rate=rate).save(output_path)

        asyncio.run(run())
        return True
    except Exception:
        return False


def _render_pyttsx3_to_file(text: str, output_path: str, speed: float = 1.0) -> bool:
    """Render pyttsx3 output to a WAV file. Returns True if successful."""
    global _pyttsx3_engine
    try:
        import pyttsx3
    except ImportError:
        return False
    if not text or not text.strip():
        return True
    try:
        with _pyttsx3_lock:
            if _pyttsx3_engine is None:
                _pyttsx3_engine = pyttsx3.init()
            _pyttsx3_engine.setProperty("rate", int(150 * max(0.5, speed)))
            _pyttsx3_engine.save_to_file(text.strip(), output_path)
            _pyttsx3_engine.runAndWait()
            # Espeak may still deliver a finished-utterance callback; brief delay avoids
            # ReferenceError: weakly-referenced object no longer exists on teardown.
            time.sleep(0.08)
        return True
    except Exception:
        return False


def _play_audio_file(audio_path: str) -> bool:
    """Play an existing audio file. Returns True if successful."""
    path = str(audio_path)
    for cmd in (
        ["mpv", "--no-video", "--really-quiet", path],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
    ):
        try:
            subprocess.run(cmd, check=True, timeout=120, capture_output=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    if path.lower().endswith(".wav"):
        try:
            subprocess.run(
                ["aplay", "-q", path],
                check=True,
                timeout=120,
                capture_output=True,
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    return False


def play_audio_file(audio_path: str | Path) -> bool:
    """Play a WAV or other audio file via mpv, ffplay, or aplay (WAV only)."""
    p = Path(audio_path).expanduser()
    if not p.is_file():
        return False
    return _play_audio_file(str(p))


def _convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any playable input to WAV with ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, output_path],
            check=True,
            timeout=120,
            capture_output=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _save_audio_file(input_path: str, output_path: str) -> bool:
    """
    Save rendered TTS audio to output_path.
    If the extension differs, attempts ffmpeg conversion.
    """
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    in_ext = Path(input_path).suffix.lower()
    out_ext = out.suffix.lower()
    try:
        if out_ext and in_ext and out_ext != in_ext:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path, str(out)],
                check=True,
                timeout=120,
                capture_output=True,
            )
        else:
            shutil.copyfile(input_path, str(out))
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return False


def _apply_applio_voice(
    input_audio_path: str,
    output_audio_path: str,
    model_path: str,
    index_path: str,
) -> tuple[bool, str]:
    """
    Run external Applio/RVC conversion.
    Uses built-in default command:
      python -m rvc_python cli --input {input_wav} --output {output_wav} --model {model_path} --index {index_path} --device cpu:0
    You can override by setting APPLIO_RVC_CMD to a command template that includes:
      {input_wav} {output_wav} {model_path} {index_path}
    """
    cmd_template = os.environ.get("APPLIO_RVC_CMD", DEFAULT_APPLIO_RVC_CMD)
    try:
        cmd = cmd_template.format(
            input_wav=input_audio_path,
            output_wav=output_audio_path,
            model_path=model_path,
            index_path=index_path,
        )
        parts = shlex.split(cmd)
        if not parts:
            return False, "empty RVC command"
        if shutil.which(parts[0]) is None:
            return False, f"RVC executable not found: {parts[0]}"
        rvc_env = os.environ.copy()
        rvc_env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            parts,
            check=True,
            timeout=180,
            capture_output=True,
            text=True,
            env=rvc_env,
        )
        return True, ""
    except (KeyError, ValueError) as e:
        return False, f"invalid APPLIO_RVC_CMD template: {e}"
    except FileNotFoundError as e:
        return False, f"RVC command not found: {e}"
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        reason = stderr if stderr else f"exit code {e.returncode}"
        return False, f"RVC conversion failed: {reason}"


def _cache_key(
    text: str,
    voice: str,
    speed: float,
    use_rvc: bool,
    model_path: str | None,
    index_path: str | None,
) -> str:
    src = "|".join(
        [
            text.strip(),
            voice,
            f"{speed:.3f}",
            "rvc" if use_rvc else "norvc",
            model_path or "",
            index_path or "",
        ]
    )
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def speak(
    text: str,
    voice: str = "en-US-AriaNeural",
    use_thread: bool = True,
    rvc_model_path: str | None = None,
    rvc_index_path: str | None = None,
    output_path: str | None = None,
    speed: float = 1.4,
    cache_enabled: bool = True,
) -> bool:
    """
    Speak the given text aloud. Tries edge-tts first, then pyttsx3.
    If rvc_model_path and rvc_index_path are provided, attempts Applio/RVC
    conversion using APPLIO_RVC_CMD before playback.
    If output_path is provided, saves the final audio (post-conversion when RVC
    succeeds) to that path.
    speed controls speaking speed (1.0 = normal; 1.4 ~= 40% faster).
    cache_enabled reuses generated audio for repeated phrases to reduce latency.
    If use_thread is True (default), runs in a background thread so the caller
    is not blocked.
    Returns True when audio was rendered (and saved if requested), otherwise False.
    """
    result = [False]

    def _run():
        if not text or not text.strip():
            result[0] = False
            return

        model = Path(rvc_model_path).expanduser() if rvc_model_path else None
        idx = Path(rvc_index_path).expanduser() if rvc_index_path else None
        use_rvc = bool(model and idx and model.exists() and idx.exists())
        key = _cache_key(
            text=text,
            voice=voice,
            speed=speed,
            use_rvc=use_rvc,
            model_path=str(model) if model else None,
            index_path=str(idx) if idx else None,
        )
        cache_ext = ".wav" if use_rvc else ".mp3"
        cache_path = _cache_root / f"{key}{cache_ext}"
        if cache_enabled and cache_path.exists():
            if output_path and not _save_audio_file(str(cache_path), output_path):
                result[0] = False
                return
            _play_audio_file(str(cache_path))
            result[0] = True
            return

        src_suffix = ".mp3"
        with tempfile.NamedTemporaryFile(suffix=src_suffix, delete=False) as src_file:
            src_path = src_file.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as rvc_in_file:
            rvc_in_path = rvc_in_file.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
            out_path = out_file.name

        try:
            rendered = _render_edge_tts_to_file(text, src_path, voice=voice, speed=speed)
            if not rendered:
                rendered = _render_pyttsx3_to_file(text, wav_path, speed=speed)
                if rendered:
                    src_path = wav_path
            if not rendered:
                result[0] = False
                return

            play_path = src_path
            if use_rvc:
                input_wav = wav_path if src_path.endswith(".wav") else rvc_in_path
                if not src_path.endswith(".wav"):
                    if not _convert_to_wav(src_path, input_wav):
                        print("[TTS] Could not convert base TTS audio to WAV for RVC; using raw TTS voice.")
                        _play_audio_file(src_path)
                        return
                rvc_ok, rvc_reason = _apply_applio_voice(input_wav, out_path, str(model), str(idx))
                if rvc_ok:
                    play_path = out_path
                    print("[TTS] Applied Applio RVC voice conversion.")
                else:
                    print(f"[TTS] RVC not applied; using raw TTS voice. Reason: {rvc_reason}")

            if cache_enabled:
                with _cache_lock:
                    _cache_root.mkdir(parents=True, exist_ok=True)
                    _save_audio_file(play_path, str(cache_path))

            if output_path and not _save_audio_file(play_path, output_path):
                result[0] = False
                return

            _play_audio_file(play_path)
            result[0] = True
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
            Path(rvc_in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    if use_thread:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return True
    else:
        _run()
        return result[0]
