# Caption Stabilizer (Commit Logic)

Reduces flicker in streaming captions by committing text only after it has been **stable for N consecutive decoder updates** (common-prefix + N-stable-updates).

## Interface

```python
from voice_recognition.stabilizer import CaptionStabilizer

stabilizer = CaptionStabilizer(stable_n=2)

# Each time the decoder returns a new partial (e.g. every 250 ms):
display_text = stabilizer.update(partial_from_decoder)

# Optional: force-commit at end of utterance
stabilizer.commit()

# Optional: new segment
stabilizer.reset()
```

**Input:** `partial` — current best hypothesis string from CTC decoder.  
**Output:** String to show on screen (committed + tentative). Committed part only updates when tentative has been unchanged for `stable_n` updates.

## Implementation Plan (Simplest Correct)

1. **State:** `committed` (final text), `last_tentative` (pending suffix), `stable_count` (how many times current tentative has been seen).
2. **On `update(partial)`:**
   - If `partial` extends `committed` (or committed is empty):  
     `tentative = partial[len(committed):]`.  
     If `tentative == last_tentative`: increment `stable_count`; if `stable_count >= stable_n` and tentative non-empty, commit it (set `committed = partial`, clear tentative).  
     Else: set `last_tentative = tentative`, `stable_count = 1`.  
     Return `committed + tentative`.
   - If `partial` does not extend `committed`: keep `committed`, clear tentative state, return `committed` (no flicker from backtracking).
3. **`commit()`:** Append current tentative to committed and clear tentative.
4. **`reset()`:** Clear committed and tentative for a new segment.

## Integration with Pipeline

```
audio capture -> mel frontend -> model -> decoder -> stabilizer -> display
                                              ^           ^
                                              |           +-- CaptionStabilizer.update(partial)
                                              +-- CTCPrefixBeamSearch.decode(log_probs) -> partial
```

Typical loop (every 250 ms):

```python
log_probs = model(mel_chunk)
partial, _ = decoder.decode(log_probs)
display_text = stabilizer.update(partial)
# show display_text on screen
```

## Jetson Orin Nano: Performance

- **Cost:** Stabilizer is pure Python (no numpy); a few string operations per update. Negligible vs decoder/model (< 0.1 ms per update).
- **Pitfalls:** None specific; keep `stable_n` small (2–3). Larger `stable_n` = less flicker but slower commit; 2 is a good default for 250 ms updates.
- **Real-time:** No extra threads or buffers; single-threaded update is fine. Call `update()` from the same thread that runs decoder/display.

## Run Tests

```bash
PYTHONPATH=src python tests/test_caption_stabilizer.py
```
