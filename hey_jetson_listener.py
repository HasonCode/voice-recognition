"""
Listen for "hey jetson" (fuzzy + regex) in the live transcript, then capture
everything said after it until there is a pause in speech. The response is
printed, optionally saved, and read aloud with AI voice (TTS).

Uses the same live pipeline as pipeline_live_test; the wake-word logic runs
on the streamed transcript. TTS uses edge-tts if available, else pyttsx3.

Usage:
  python hey_jetson_listener.py
  python hey_jetson_listener.py --output responses.txt
  python hey_jetson_listener.py --no-voice              # disable TTS
  python hey_jetson_listener.py --voice-id en-US-GuyNeural
  pip install edge-tts   # for AI-style voice (needs internet to generate)
  pip install pyttsx3    # offline fallback (e.g. espeak on Linux)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voice_recognition.audio.config import AudioConfig
from voice_recognition.decoder import CTCPrefixBeamSearch
from voice_recognition.pipeline import StreamingCaptionPipeline, StreamingConfig
from voice_recognition.postprocess import merge_display_into_transcript
from voice_recognition.stabilizer import CaptionStabilizer
from voice_recognition.wakeword import HeyJetsonListener, speak

NEMO_MODEL_PATH = Path(__file__).resolve().parent / "src" / "voice_recognition" / "models" / "ctc_small.nemo"
VOCAB_DUMMY = ["<blank>", "a", "b", "c", " "]
BLANK_DUMMY = 0


def make_dummy_model(vocab_size=len(VOCAB_DUMMY), bias_blank=True):
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


def main(
    use_nemo=True,
    device=None,
    output_path=None,
    pause_sec=2.5,
    similarity_threshold=0.75,
    voice=True,
    voice_id="en-US-AriaNeural",
):
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
    last_display = [None]
    identical_count = [0]
    STUCK_THRESHOLD = 8
    pipeline_ref = [None]

    def on_response(response: str):
        """Called when user said 'hey jetson' and then spoke until a pause."""
        if not response.strip():
            return
        print("\n[Response]", response)
        if output_path:
            path = Path(output_path)
            with open(path, "a", encoding="utf-8") as f:
                f.write(response.strip() + "\n")
            print(f"  (appended to {path})")
        if voice:
            speak(response, voice=voice_id)

    listener = HeyJetsonListener(
        on_response=on_response,
        similarity_threshold=similarity_threshold,
        pause_sec=pause_sec,
    )

    def on_display(s):
        if s == last_display[0]:
            identical_count[0] += 1
            if identical_count[0] == STUCK_THRESHOLD and pipeline_ref[0] is not None:
                pipeline_ref[0].stabilizer.reset()
            return
        identical_count[0] = 0
        last_display[0] = s
        accumulated_transcript[0] = merge_display_into_transcript(accumulated_transcript[0], s)
        print("display:", repr(s))
        listener.push_transcript(accumulated_transcript[0])

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

    print(f"Running with {model_name} model. Say 'Hey Jetson' then your question; pause when done.")
    if voice:
        print("AI voice will read back each captured response.")
    if output_path:
        print(f"Responses will be appended to {output_path}\n")
    try:
        pipeline.run(device=device)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        listener.stop()
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Listen for 'Hey Jetson', capture speech until pause.")
    p.add_argument("--output", "-o", type=str, help="Append each response to this file")
    p.add_argument("--device", "-d", type=int, default=None, help="Microphone device index")
    p.add_argument("--pause", "-p", type=float, default=2.5, help="Seconds of silence to end capture")
    p.add_argument("--similarity", "-s", type=float, default=0.75, help="Fuzzy match threshold for 'hey jetson' (0–1)")
    p.add_argument("--dummy", action="store_true", help="Use dummy model")
    p.add_argument("--no-voice", action="store_true", help="Disable TTS reading of responses")
    p.add_argument("--voice-id", type=str, default="en-US-AriaNeural", help="Edge TTS voice (e.g. en-US-GuyNeural)")
    args = p.parse_args()
    main(
        use_nemo=not args.dummy,
        device=args.device,
        output_path=args.output,
        pause_sec=args.pause,
        similarity_threshold=args.similarity,
        voice=not args.no_voice,
        voice_id=args.voice_id,
    )
