# CTC Prefix Beam Search Decoder

Standalone CTC prefix beam search (beam=8, no LM) for edge streaming ASR.

## Interface

```python
from voice_recognition.decoder import CTCPrefixBeamSearch

vocab = ["<blank>", "a", "b", "c", " ", ...]  # Match model vocab
decoder = CTCPrefixBeamSearch(vocab, blank_index=0, beam_size=8)

# log_probs: (T, V) from model.forward() or ONNX inference
text, log_score = decoder.decode(log_probs)
```

**Input:** `log_probs` shape `(T, V)` — log probabilities per frame (numpy or torch).  
**Output:** Best hypothesis string and its log-score.

## Integration with Pipeline

```
audio capture -> mel frontend -> model -> decoder -> stabilizer -> display
                      ^              ^         ^
                      |              |         +-- CTCPrefixBeamSearch.decode()
                      |              +-- CTC log-probs (T, V)
                      +-- 80-bin log-mel, 25ms/10ms, FFT 512
```

1. **Mel frontend** — existing `MelFeatureExtractor` with 1.6s rolling context, 250ms update stride.
2. **Model** — CTC ASR (PyTorch/ONNX/TensorRT) outputs `(T, V)` log-probs.
3. **Decoder** — `CTCPrefixBeamSearch.decode(log_probs)` → text.
4. **Stabilizer** — common-prefix + N-stable-updates to reduce flicker before display.

## Jetson Orin Nano: Performance Pitfalls

### 1. **Python overhead**
- Decoder is pure Python + numpy; ~1–5 ms per decode for typical T=16–25.
- On Jetson, keep decode batches small; avoid per-frame decode if T is very short.
- **Mitigation:** Decode every 250 ms (T≈25 frames); aggregate model inference.

### 2. **Memory bandwidth**
- NumPy ops on CPU compete with GPU for shared memory.
- **Mitigation:** Prefer `model_output.cpu().numpy()` once per chunk; avoid repeated CPU↔GPU sync.

### 3. **Model inference latency**
- CTC model (e.g. conformer-lite) is usually the bottleneck, not the decoder.
- **Mitigation:** ONNX/TensorRT, FP16, and batching where possible; avoid dynamic shapes.

### 4. **JetPack 5 / Python 3.8**
- Use `numpy` wheels from `pip` or NVIDIA containers; avoid building from source.
- Compatible with PyTorch 2.x from NVIDIA’s Jetson PyTorch wheels.

### 5. **Real-time budget**
- Target: 250 ms update → decode must stay well under 50 ms.
- Beam size 8 is usually fine; avoid beam > 16 for real-time.

## Run Tests

```bash
PYTHONPATH=src python tests/test_ctc_prefix_beam_search.py
```
