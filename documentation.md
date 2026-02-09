# Voice Recognition Model — Documentation

## Overview

This project provides a working directory for a voice recognition model, with an initial focus on **audio collection** and feature extraction following specified encoding standards.

---

## Encoding Standards

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Audio** | Mono 16 kHz | Single channel, 16,000 samples/second |
| **Features** | 80-bin log-Mel filterbanks | Mel-scale spectrogram, log-compressed |
| **STFT window** | 25 ms | 400 samples at 16 kHz |
| **STFT hop** | 10 ms | 160 samples at 16 kHz |
| **FFT size** | 512 | Power-of-two for efficiency |
| **Streaming** | Ring buffer | Continuous computation into fixed-size buffer |
| **Normalization** | Online CMVN | Sliding mean/variance over ~3 seconds (streaming) |

---

## Project Structure

```
voice_recognition/
├── pyproject.toml           # Dependencies, scripts, package config
├── documentation.md         # This file
├── .gitignore
├── tests/
│   ├── test_ctc_prefix_beam_search.py
│   ├── test_caption_stabilizer.py
│   └── test_streaming_loop.py
└── src/
    └── voice_recognition/
        ├── __init__.py
        ├── cli.py           # voice-record CLI
        ├── audio/
        │   ├── __init__.py
        │   ├── config.py    # AudioConfig (encoding standards)
        │   ├── collector.py # AudioCollector (mono 16 kHz recording)
        │   └── features.py  # MelFeatureExtractor, RingBuffer, online CMVN
        ├── decoder/
        │   ├── __init__.py
        │   ├── README.md    # Jetson pitfalls, integration
        │   └── ctc_prefix_beam_search.py  # CTC prefix beam search (beam=8, no LM)
        ├── stabilizer/
        │   ├── __init__.py
        │   ├── README.md    # Integration, Jetson notes
        │   └── caption_stabilizer.py  # N-stable commit logic for streaming captions
        ├── postprocess.py            # refine_bpe_caption (BPE/subword → readable)
        ├── pipeline/
        │   ├── __init__.py
        │   ├── README.md    # Jetson pitfalls, integration
        │   └── streaming_loop.py     # End-to-end streaming loop glue
        └── models/
            ├── __init__.py
            ├── ctc_small.nemo       # NeMo CTC ASR checkpoint
            └── nemo_model.py        # NeMo model loader
```

---

## Changes Log

### 1. Audio Configuration (`src/voice_recognition/audio/config.py`)

- **Added** `AudioConfig` dataclass with all encoding standards:
  - `sample_rate=16000`, `channels=1` (mono)
  - `window_ms=25`, `hop_ms=10`, `fft_size=512`
  - `n_mels=80`
  - `cmvn_window_sec=3.0` for online CMVN sliding window
- **Computed properties:** `frame_length`, `hop_length`, `frames_per_second`, `cmvn_window_frames`

### 2. Audio Collector (`src/voice_recognition/audio/collector.py`)

- **Added** `AudioCollector` for mono 16 kHz recording:
  - `record_chunk(duration_sec)` — single batch recording
  - `record_stream(chunk_duration_sec)` — streaming chunks as iterator
  - `record_to_file(filepath, duration_sec)` — record and save WAV
- **Uses** `sounddevice` for capture; resampling to 16 kHz if needed is left to caller (input expected at 16 kHz via device/config)

### 3. Feature Extraction (`src/voice_recognition/audio/features.py`)

- **Added** `RingBuffer` — fixed-size ring buffer for streaming audio/features
- **Added** `MelFeatureExtractor`:
  - `stft(audio)` — STFT power spectrum (25 ms / 10 ms hop / FFT 512)
  - `power_to_mel(power_spec)` — 80-bin log-Mel filterbanks
  - `extract(audio)` — full pipeline: STFT → Mel → log
  - `apply_online_cmvn(features, frame_buffer)` — sliding mean/var normalization
  - `extract_streaming(audio_chunk, audio_ring, frame_ring)` — streaming extraction with ring buffers and online CMVN

### 4. CLI (`src/voice_recognition/cli.py`)

- **Added** `voice-record` entry point:
  - `--duration` — recording length in seconds (default: 5)
  - `--output` / `-o` — output WAV path
  - `--device` — input device index
  - `--list-devices` — list audio devices and exit
  - `--extract-features` — extract log-Mel features after recording

### 5. Dependencies (`pyproject.toml`)

- **Core:** `numpy`, `scipy`, `sounddevice`
- **Optional ML:** `torch`, `torchvision` (install with `pip install -e ".[ml]"`)
- **CLI script:** `voice-record` registered in `[project.scripts]`
- **Package layout:** `[tool.poetry.packages]` with `src/` layout

### 6. `.gitignore`

- **Added:** `*.wav`, `*.mp3`, `__pycache__/`, `*.egg-info/`, `.venv/`, `venv/` to ignore audio artifacts and Python build artifacts

### 7. CTC Prefix Beam Search Decoder (`src/voice_recognition/decoder/`)

- **Added** `CTCPrefixBeamSearch` — CTC prefix beam search (beam=8, no LM)
  - Interface: `decode(log_probs) -> (text, score)`
  - Input: (T, V) log-probs from CTC model; accepts numpy or torch
  - Minimal deps: numpy only
- **Added** `ctc_greedy_decode()` for validation/comparison
- **Tests:** `tests/test_ctc_prefix_beam_search.py` — unit tests and toy example
- **Integration:** `decoder/README.md` — pipeline placement, Jetson pitfalls, real-time tips

### 8. Caption Stabilizer (`src/voice_recognition/stabilizer/`)

- **Added** `CaptionStabilizer` — common-prefix + N-stable-updates commit logic
  - `update(partial)` → display string (committed + tentative)
  - `commit()` — force-commit current tentative; `reset()` — new segment
  - No new dependencies (stdlib only)
- **Tests:** `tests/test_caption_stabilizer.py` — unit tests and toy example
- **Integration:** `stabilizer/README.md` — pipeline placement, Jetson performance notes

### 9. Streaming Pipeline (`src/voice_recognition/pipeline/`)

- **Added** `StreamingConfig` — context_sec=1.6, update_interval_sec=0.25
- **Added** `StreamingCaptionPipeline` — glue: audio → mel/audio → model → decoder → stabilizer → display
  - `run(audio_iterator=None)` — live mic or injected iterator; `stop()` to exit
  - `run_for_n_updates(n, audio_iterator)` — for tests; returns list of display strings
  - `model_input="mel"` (default) or `"audio"` — pass mel features or raw audio to model_forward
  - Model: callable `(mel or audio) -> log_probs: (T, V)`
- **Tests:** `tests/test_streaming_loop.py` — unit tests and toy example
- **Integration:** `pipeline/README.md` — Jetson pitfalls, real-time notes

### 10. BPE Post-Processor (`src/voice_recognition/postprocess.py`)

- **Added** `refine_bpe_caption(raw)` — converts raw BPE/subword output to readable text
  - Removes `<?>` placeholders, `##` WordPiece continuation, `▁` SentencePiece boundary
  - Collapses multiple spaces
- **Pipeline integration:** Applied between decoder and stabilizer; configurable via `post_process` parameter (default: `refine_bpe_caption`)

### 11. NeMo Model Support (`src/voice_recognition/models/`)

- **Added** `load_nemo_model(path)` — loads `.nemo` checkpoint, returns `(model_forward, vocab, blank_index)`
  - NeMo models expect raw audio; use `model_input="audio"` in the pipeline
  - Model location: `src/voice_recognition/models/ctc_small.nemo`
- **pipeline_test.py** — run pipeline with NeMo or dummy:
  - `python pipeline_test.py` — NeMo model, fake audio
  - `python pipeline_test.py --dummy` — dummy model
  - `python pipeline_test.py --file test.wav` — NeMo model, audio from WAV (16 kHz mono)

---

## Usage

### Install

```bash
cd voice_recognition
pip install -e .
```

### Record audio

```bash
voice-record --duration 5 --output sample.wav
voice-record -o sample.wav --duration 10 --extract-features
voice-record --list-devices
```

### Programmatic usage

```python
from voice_recognition.audio import AudioCollector, MelFeatureExtractor
from voice_recognition.audio.config import AudioConfig

config = AudioConfig()

# Record
collector = AudioCollector(config)
audio = collector.record_chunk(5.0)
# Or: collector.record_to_file("sample.wav", 5.0)

# Extract features (batch)
extractor = MelFeatureExtractor(config)
features = extractor.extract(audio)  # (n_frames, 80)

# Streaming with ring buffers and online CMVN
audio_ring, frame_ring = None, None
for chunk in collector.record_stream(chunk_duration_sec=0.1):
    features, audio_ring, frame_ring = extractor.extract_streaming(
        chunk, audio_ring, frame_ring, apply_cmvn=True
    )
    # Process features...
```

---

## Next Steps

- Dataset collection pipeline (batch recording, labeling)
- Model architecture (e.g., conformer / transformer for streaming ASR)
- Training loop and evaluation
- Export for Jetson Orin Nano deployment
