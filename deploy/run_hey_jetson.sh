#!/bin/bash
# Run Hey Jetson listener (for systemd or manual use).
# Set VOICE_RECOGNITION_ROOT to the project root, or we use the directory of this script.
set -e
ROOT="${VOICE_RECOGNITION_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi
# Use --device 0 for default mic; change if your mic is different.
exec python hey_jetson_listener.py --device 0
