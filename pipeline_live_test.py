"""Live audio transcription using the streaming pipeline and batched microphone input.

Uses the same pipeline as file mode: 250 ms chunks, 1.6 s rolling context,
ring buffer, model -> decoder -> stabilizer -> display. Audio is read from
the default (or specified) microphone in batches of update_interval_sec.

Usage:
  python pipeline_live_test.py                    # NeMo model, default mic
  python pipeline_live_test.py --output out.txt   # Write transcript to file as it updates
  python pipeline_live_test.py --device 1          # Use microphone device 1
  python pipeline_live_test.py --dummy             # Dummy model (no NeMo)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voice_recognition.audio.config import AudioConfig
from voice_recognition.decoder import CTCPrefixBeamSearch
from voice_recognition.pipeline import StreamingCaptionPipeline, StreamingConfig
from voice_recognition.postprocess import merge_display_into_transcript
from voice_recognition.stabilizer import CaptionStabilizer

NEMO_MODEL_PATH = Path(__file__).resolve().parent / "src" / "voice_recognition" / "models" / "ctc_small.nemo"
VOCAB_DUMMY = ["<blank>", "a", "b", "c", " "]
BLANK_DUMMY = 0


def make_dummy_model(vocab_size=len(VOCAB_DUMMY), bias_blank=True):
    """Dummy model for testing without NeMo."""
    import numpy as np

    def forward(mel):
        T = mel.shape[0]
        V = vocab_size
        log_probs = np.random.randn(T, V).astype(np.float32) * 0.1
        if bias_blank:
            log_probs[:, BLANK_DUMMY] += 1.0
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))
        return log_probs

    return forward


def main(use_nemo=True, output_path=None, device=None):
    config = StreamingConfig(context_sec=1.6, update_interval_sec=0.25)

    if use_nemo and NEMO_MODEL_PATH.exists():
        from voice_recognition.models import load_nemo_model

        print(f"Loading NeMo model from {NEMO_MODEL_PATH}...")
        model_forward, vocab, blank_index = load_nemo_model(NEMO_MODEL_PATH)
        decoder = CTCPrefixBeamSearch(vocab, blank_index=blank_index, beam_size=8)
        model_input = "audio"
        model_name = "NeMo"
    else:
        if use_nemo:
            print(f"NeMo model not found at {NEMO_MODEL_PATH}, using dummy model.")
        model_forward = make_dummy_model()
        decoder = CTCPrefixBeamSearch(VOCAB_DUMMY, blank_index=BLANK_DUMMY, beam_size=8)
        model_input = "mel"
        model_name = "dummy"

    accumulated_transcript = [""]
    written_to_file = [""]  # prefix of accumulated already appended to output file
    last_display = [None]  # only update when display actually changes
    identical_count = [0]  # consecutive identical displays (to unstick)
    STUCK_THRESHOLD = 8  # reset stabilizer after this many identical displays
    pipeline_ref = [None]  # set after construction so on_display can reset stabilizer

    def _append_to_output(acc: str, force_new_display=None):
        """Append only the new part of the transcript so the file is a continuous stream."""
        if not output_path:
            return
        prev = written_to_file[0]
        # When merge didn't extend (no overlap), we may have been given the new display to append
        if force_new_display:
            to_append = force_new_display.strip()
            if to_append:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(" " + to_append if prev else to_append)
                written_to_file[0] = (prev + " " + to_append) if prev else to_append
            return
        if acc.startswith(prev):
            new_part = acc[len(prev) :].lstrip()
            if new_part:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(" " + new_part if prev else new_part)
                written_to_file[0] = acc
        elif prev.startswith(acc):
            # We've already written more than current accumulated (e.g. previous no-overlap append); skip
            pass
        else:
            # New segment (e.g. stabilizer reset)
            with open(output_path, "a", encoding="utf-8") as f:
                if prev:
                    f.write("\n")
                f.write(acc)
            written_to_file[0] = acc

    def on_display(s):
        if s == last_display[0]:
            identical_count[0] += 1
            # Unstick: after many identical batches, reset stabilizer so next batch can update
            if identical_count[0] == STUCK_THRESHOLD and pipeline_ref[0] is not None:
                pipeline_ref[0].stabilizer.reset()
            return
        identical_count[0] = 0
        last_display[0] = s
        old_acc = accumulated_transcript[0]
        accumulated_transcript[0] = merge_display_into_transcript(old_acc, s)
        print("display:", repr(s))
        # If merge didn't extend (no word overlap, kept longer), append the new display anyway
        if accumulated_transcript[0] == old_acc and s and s.strip():
            _append_to_output(accumulated_transcript[0], force_new_display=s)
        else:
            _append_to_output(accumulated_transcript[0])

    pipeline = StreamingCaptionPipeline(
        config=config,
        audio_config=AudioConfig(),
        model_forward=model_forward,
        model_input=model_input,
        decoder=decoder,
        stabilizer=CaptionStabilizer(stable_n=2),
        on_display=on_display,
    )
    pipeline_ref[0] = pipeline

    print(f"Running live transcription with {model_name} model (batched mic input)...")
    print("Speak into the microphone. Press Ctrl+C to stop.\n")
    # Start with empty file so this run is one continuous stream
    if output_path:
        Path(output_path).write_text("", encoding="utf-8")

    try:
        pipeline.run(device=device)
    except KeyboardInterrupt:
        print("\nStopped.")

    # Append any remaining transcript not yet written (e.g. last update)
    if output_path:
        _append_to_output(accumulated_transcript[0])
        print(f"Transcription saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    args = sys.argv[1:]
    use_nemo = "--dummy" not in args
    output_path = "transcription.txt"
    device = None
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_path = args[idx + 1]
    if "--device" in args:
        idx = args.index("--device")
        if idx + 1 < len(args):
            try:
                device = int(args[idx + 1])
            except ValueError:
                pass
    main(use_nemo=use_nemo, output_path=output_path, device=device)
