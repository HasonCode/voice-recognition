"""CLI for audio collection."""

import argparse
import sys
from pathlib import Path

from voice_recognition.audio import AudioCollector, MelFeatureExtractor
from voice_recognition.audio.config import AudioConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Record audio for voice recognition (mono 16 kHz)")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("recording.wav"),
        help="Output WAV file path",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (list with --list-devices)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Extract and print log-Mel features after recording",
    )
    args = parser.parse_args()

    if args.list_devices:
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except ImportError:
            print("sounddevice not installed: pip install sounddevice", file=sys.stderr)
            sys.exit(1)
        return

    config = AudioConfig()
    collector = AudioCollector(config)

    print(f"Recording {args.duration}s to {args.output} (mono {config.sample_rate} Hz)...")
    collector.record_to_file(str(args.output), args.duration, args.device)
    print(f"Saved: {args.output}")

    if args.extract_features:
        import numpy as np
        import scipy.io.wavfile as wavfile

        sr, audio = wavfile.read(str(args.output))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768
        extractor = MelFeatureExtractor(config)
        features = extractor.extract(audio)
        print(f"Extracted {features.shape[0]} frames x {features.shape[1]} Mel bins")
        if features.shape[0] > 0:
            print(f"Sample frame (first 5 bins): {features[0, :5]}")


if __name__ == "__main__":
    main()
