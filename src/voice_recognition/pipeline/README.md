# Streaming Caption Pipeline (Glue)

End-to-end loop: **audio → mel → model → decoder → stabilizer → display**.

## Implementation Plan

1. **Audio ring buffer:** Hold the last `context_sec` (e.g. 1.6 s) of samples.
2. **Update cadence:** Every `update_interval_sec` (e.g. 250 ms), push one chunk of new audio and run the rest of the pipeline.
3. **Per update:** Take the last `context_sec` of audio → extract 80-bin log-mel (batch) → `model_forward(mel)` → `decoder.decode(log_probs)` → `stabilizer.update(partial)` → `on_display(text)`.
4. **Audio source:** Either live microphone via `AudioCollector.record_stream()` or an injected iterator (e.g. for tests or file playback).

No new dependencies; uses existing audio, features, decoder, and stabilizer.

## Interface

```python
from voice_recognition.pipeline import StreamingCaptionPipeline, StreamingConfig
from voice_recognition.audio import MelFeatureExtractor, AudioCollector
from voice_recognition.audio.config import AudioConfig
from voice_recognition.decoder import CTCPrefixBeamSearch
from voice_recognition.stabilizer import CaptionStabilizer

config = StreamingConfig(context_sec=1.6, update_interval_sec=0.25)
audio_config = AudioConfig()
mel_extractor = MelFeatureExtractor(audio_config)
decoder = CTCPrefixBeamSearch(vocab, blank_index=0, beam_size=8)
stabilizer = CaptionStabilizer(stable_n=2)

def model_forward(mel: np.ndarray) -> np.ndarray:
    # Your PyTorch / ONNX / TensorRT: (T, 80) -> (T, V) log-probs
    return log_probs

pipeline = StreamingCaptionPipeline(
    config=config,
    audio_config=audio_config,
    mel_extractor=mel_extractor,
    model_forward=model_forward,
    decoder=decoder,
    stabilizer=stabilizer,
    on_display=print,
)

# Live mic (blocks until stop())
pipeline.run()
# Or with injected audio (e.g. tests):
# pipeline.run(audio_iterator=iter(list_of_chunks))
```

**Model contract:** `model_forward(mel: np.ndarray (T, 80)) -> log_probs: np.ndarray (T, V)`.

## Jetson Orin Nano: Performance and Pitfalls

| Pitfall | Mitigation |
|--------|------------|
| **Single-thread latency** | The loop is synchronous: one 250 ms period = read chunk + mel + model + decode + stabilizer. Model (and possibly mel) dominate; keep inference &lt; ~200 ms so the loop keeps up. |
| **CPU/GPU sync** | Call `model_forward` with numpy; if the model is on GPU, do a single `.cpu().numpy()` inside `model_forward` and avoid extra syncs elsewhere. |
| **Buffer size** | 1.6 s × 16 kHz = 25 600 samples; small. Avoid copying large buffers; ring buffer overwrites in place. |
| **Blocking mic** | `record_stream()` blocks on the next chunk. Run the whole pipeline on one thread, or run audio in a separate thread and feed chunks via a queue (then the loop pulls from the queue). |
| **Real-time** | To keep up with 250 ms updates, ensure: mel + model + decode + stabilizer &lt; 250 ms. On Jetson, model (ONNX/TensorRT, FP16) is usually the only heavy part. |

## Run Tests

```bash
PYTHONPATH=src python tests/test_streaming_loop.py
```
