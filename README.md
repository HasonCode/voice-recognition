# Voice Recognition

A model for **edge-based voice identification and transcription** of audio. Designed to run on resource-constrained devices (e.g., NVIDIA Jetson Orin Nano) for on-device speech recognition without requiring cloud connectivity.

## Aims

- **Voice identification** — recognize who is speaking
- **Transcription** — convert speech to text
- **Edge deployment** — run inference locally with low latency and privacy-preserving processing

## Current Status

Initial development focuses on **audio collection** and feature extraction with streaming support. The pipeline uses mono 16 kHz audio, 80-bin log-Mel filterbanks, and online CMVN suitable for real-time inference.

See [documentation.md](documentation.md) for encoding standards, API details, and usage.

## Quick Start

```bash
pip install -e .
voice-record --duration 5 -o sample.wav
```
