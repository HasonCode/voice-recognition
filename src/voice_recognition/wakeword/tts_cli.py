"""CLI utility to test TTS output quickly."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from voice_recognition.wakeword.tts import speak


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = _project_root()
    default_rvc_model = root / "hason_voice_200e_22800s.pth"
    default_rvc_index = root / "hason_voice.index"

    parser = argparse.ArgumentParser(
        description="Test text-to-speech with optional Applio/RVC voice conversion."
    )
    parser.add_argument(
        "--text",
        "-t",
        default="Hello. This is a text to speech test.",
        help="Text to speak.",
    )
    parser.add_argument(
        "--voice-id",
        default="en-US-AriaNeural",
        help="Edge TTS voice id (for example: en-US-GuyNeural).",
    )
    parser.add_argument(
        "--rvc-model",
        type=str,
        default=str(default_rvc_model),
        help="Path to Applio/RVC .pth model.",
    )
    parser.add_argument(
        "--rvc-index",
        type=str,
        default=str(default_rvc_index),
        help="Path to Applio/RVC .index file.",
    )
    parser.add_argument(
        "--no-rvc",
        action="store_true",
        help="Disable Applio/RVC conversion and use raw TTS playback only.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output audio file path to save synthesized speech.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.4,
        help="Speech speed multiplier (1.0 = normal, 1.4 = 40%% faster).",
    )
    args = parser.parse_args()

    model_path = None if args.no_rvc else args.rvc_model
    index_path = None if args.no_rvc else args.rvc_index

    if not args.no_rvc:
        if not Path(args.rvc_model).expanduser().exists():
            print(f"[TTS test] RVC model not found: {args.rvc_model}")
        if not Path(args.rvc_index).expanduser().exists():
            print(f"[TTS test] RVC index not found: {args.rvc_index}")
        if not os.environ.get("APPLIO_RVC_CMD"):
            print("[TTS test] APPLIO_RVC_CMD not set; using built-in default RVC command.")
            if shutil.which("rvc-python") is None:
                print("[TTS test] rvc-python executable not found in PATH.")

    print(f"[TTS test] Speaking: {args.text}")
    ok = speak(
        args.text,
        voice=args.voice_id,
        use_thread=False,
        rvc_model_path=model_path,
        rvc_index_path=index_path,
        output_path=args.output,
        speed=args.speed,
    )
    if args.output:
        abs_out = str(Path(args.output).expanduser().resolve())
        if ok and Path(abs_out).exists():
            print(f"[TTS test] Saved: {abs_out}")
        else:
            print(f"[TTS test] Save failed: {abs_out}")
    print("[TTS test] Done.")


if __name__ == "__main__":
    main()
