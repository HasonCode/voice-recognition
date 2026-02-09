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

## Llama 3.2 3B Instruct (transcript processing)

To feed live or saved transcripts into **Llama 3.2 3B Instruct** (summarization, Q&A, etc.):

### 1. Access the model

- The model is **gated** on Hugging Face. You must:
  - Create a Hugging Face account and accept [Meta’s Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
  - Log in: `pip install huggingface_hub` then `huggingface-cli login` (use a token from [HF Settings → Access Tokens](https://huggingface.co/settings/tokens)).

### 2. Install dependencies

From the project root:

```bash
pip install transformers accelerate
# Optional, for 4-bit to reduce VRAM (~2–3 GB):
pip install bitsandbytes
```

Or add to `pyproject.toml` under `[project.optional-dependencies]`:

```toml
llama = [
    "transformers>=4.45.0",
    "accelerate>=0.33.0",
    "bitsandbytes>=0.44.0",  # optional, for load_in_4bit
]
```

Then: `pip install -e ".[llama]"`.

### 3. Run the example script

Use the provided script to load the model and run on a transcript (e.g. from a file or passed as text):

```bash
python llama_transcript_example.py "Your transcript text here"
# Or with a file:
python llama_transcript_example.py --file transcription.txt
```

See `llama_transcript_example.py` for the exact load/generate code and how to plug in your own transcript source.

---

## Llama on the edge

You can run Llama (including 3.2 3B Instruct) **on-device** (Jetson Orin Nano, Raspberry Pi, other ARM/x86 edge devices) without a GPU or with limited VRAM by using **llama.cpp** with **GGUF** models. This fits a pipeline where transcript is produced on the same device and you want LLM processing locally.

### Option 1: llama.cpp + GGUF (recommended for edge)

- **What it is:** C++ inference engine, CPU-first (ARM/NEON supported), with optional CUDA on Jetson. Uses quantized GGUF models (Q4_K_M ~2 GB for 3B).
- **Where it runs:** Jetson Orin Nano, Jetson Nano, Raspberry Pi 5 (4–8 GB RAM), other ARM or x86 boards.
- **Steps:**
  1. **Get a GGUF model** (no Hugging Face gating for many community quantized builds):
     - [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) — e.g. `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~2 GB).
     - Or search Hugging Face for `Llama-3.2-3B-Instruct-GGUF`.
  2. **Install llama.cpp** (build from source on ARM/Jetson for your platform):
     - Clone [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp), then `cmake -B build && cmake --build build`. On Jetson you can enable CUDA in the build for GPU acceleration.
  3. **Run the server or CLI:**
     - Server: `./build/bin/server -m path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf -c 2048 --port 8080`
     - Then send transcript to `http://localhost:8080/completion` (or use the OpenAI-compatible endpoint if available).

### Option 2: llama-cpp-python (Python API, same GGUF)

- **What it is:** Python bindings for llama.cpp. Same GGUF files; you load the model in Python and call `generate()` or chat APIs so you can feed transcript from your voice pipeline in process.
- **Install:**
  ```bash
  pip install llama-cpp-python
  # On Jetson with CUDA (optional):
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```
- **Use:** Load the `.gguf` file with `Llama(model_path="...")` and run prompts (see `llama_transcript_edge_example.py`).

### Option 3: TensorRT-LLM (Jetson AGX Orin only)

- **What it is:** NVIDIA’s optimized LLM runtime. Best throughput on Jetson when supported.
- **Caveat:** TensorRT-LLM is validated for **Jetson AGX Orin** (e.g. JetPack 6.1, L4T r36.4). **Jetson Orin Nano** is not officially supported (no planned support in current issues). Prefer llama.cpp/GGUF for Orin Nano.

### Option 4: Ollama (easiest if your device is supported)

- **What it is:** One-command local LLM runner; it downloads and runs models (many in GGUF under the hood).
- **Edge:** Supports ARM64 (e.g. Raspberry Pi, some Jetson setups). Check [ollama.com](https://ollama.com) for your OS/board.
  ```bash
  ollama run llama3.2:3b-instruct
  ```
- Then call the local API with your transcript (e.g. `POST /api/generate`).

### Summary

| Target device        | Suggested stack              |
|----------------------|-----------------------------|
| Jetson Orin Nano     | llama.cpp + GGUF (CPU or CUDA build), or llama-cpp-python |
| Jetson AGX Orin      | TensorRT-LLM or llama.cpp   |
| Raspberry Pi 5 (4–8 GB) | llama.cpp + Q4_K_M GGUF, or Ollama |
| Any edge (low RAM)   | 1B GGUF or smaller quantizations (Q3, IQ3) |

Use `llama_transcript_edge_example.py` to feed transcript into a local GGUF model via llama-cpp-python from this repo.

---

## Deploy on Jetson Orin Nano (JetPack 5)

JetPack 5 uses **Python 3.8**; the project supports 3.8 and the deploy guide assumes it. To run the **Hey Jetson** listener as a readily listening bot:

1. Copy the project onto the device and set up a venv with Python 3.8 (`python3.8 -m venv .venv`).
2. Install dependencies (see full guide; NeMo/sounddevice/pyttsx3 or edge-tts; optional `docs/requirements-jetson38.txt` for 3.8).
3. Test: `python hey_jetson_listener.py --device 0`.
4. Run at boot: use the **systemd service** and run script in `deploy/`:
   - `deploy/run_hey_jetson.sh` — activates venv and runs the listener.
   - `deploy/hey-jetson.service` — systemd unit; copy to `/etc/systemd/system/`, edit user/path, then `enable` and `start`.

Full steps, paths, and optional TTS/Llama notes: **[docs/DEPLOY_JETSON_ORIN_NANO.md](docs/DEPLOY_JETSON_ORIN_NANO.md)**.

---

## Next Steps

- Dataset collection pipeline (batch recording, labeling)
- Model architecture (e.g., conformer / transformer for streaming ASR)
- Training loop and evaluation
- Export for Jetson Orin Nano deployment
