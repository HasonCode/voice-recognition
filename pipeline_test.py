"""Run the full streaming pipeline with NeMo or dummy model.

Usage:
  python pipeline_test.py              # NeMo model, fake audio
  python pipeline_test.py --dummy      # Dummy model, fake audio
  python pipeline_test.py --file test.wav   # NeMo model, audio from file
  python pipeline_test.py --file test.wav --output transcript.txt   # Save transcription
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np

from voice_recognition.audio import MelFeatureExtractor
from voice_recognition.audio.config import AudioConfig
from voice_recognition.decoder import CTCPrefixBeamSearch
from voice_recognition.pipeline import StreamingCaptionPipeline, StreamingConfig
from voice_recognition.postprocess import merge_display_into_transcript
from voice_recognition.stabilizer import CaptionStabilizer

# Path to NeMo model
NEMO_MODEL_PATH = Path(__file__).resolve().parent / "src" / "voice_recognition" / "models" / "ctc_small.nemo"

VOCAB_DUMMY = ["<blank>", "a", "b", "c", " "]
BLANK_DUMMY = 0


def load_wav_chunks(path: Path, chunk_samples: int, sample_rate: int = 16000):
    """Load WAV and yield 250 ms chunks as float32."""
    import scipy.io.wavfile as wavfile
    sr, audio = wavfile.read(str(path))
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if sr != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, got {sr} Hz. Resample the file.")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) > 0:
            yield chunk.astype(np.float32)


def make_dummy_model(vocab_size=len(VOCAB_DUMMY), bias_blank=True):
    """Dummy model: mel (T, 80) -> random log_probs (T, V)."""

    def forward(mel):
        T = mel.shape[0]
        V = vocab_size
        log_probs = np.random.randn(T, V).astype(np.float32) * 0.1
        if bias_blank:
            log_probs[:, BLANK_DUMMY] += 1.0
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))
        return log_probs

    return forward


def main(use_nemo=True, wav_path=None, output_path=None):
    config = StreamingConfig(context_sec=1.6, update_interval_sec=0.25)
    chunk_samples = config.chunk_samples

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

    accumulated_transcript = [""]  # mutable; merge all displays for full caption

    def on_display(s):
        accumulated_transcript[0] = merge_display_into_transcript(accumulated_transcript[0], s)
        print("display:", repr(s))

    pipeline = StreamingCaptionPipeline(
        config=config,
        audio_config=AudioConfig(),
        model_forward=model_forward,
        model_input=model_input,
        decoder=decoder,
        stabilizer=CaptionStabilizer(stable_n=2),
        on_display=on_display,
    )

    if wav_path:
        wav_path = Path(wav_path)
        if not wav_path.exists():
            print(f"File not found: {wav_path}")
            sys.exit(1)
        audio_source = "file"
    else:
        chunks = [
            np.random.rand(chunk_samples).astype(np.float32) * 0.1
            for _ in range(12)
        ]
        audio_source = "fake"

    print(f"Running pipeline with {model_name} model ({audio_source} audio)...\n")
    if wav_path:
        # Larger hop (0.5s) reduces overlap vs default 0.25s, fewer conflicting hypotheses
        pipeline.run_from_file(str(wav_path), hop_sec=0.5)
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(accumulated_transcript[0], encoding="utf-8")
            print(f"\nTranscription saved to {output_path}")
    else:
        pipeline.run_for_n_updates(5, iter(chunks))
    print("\nDone.")


if __name__ == "__main__":
    args = sys.argv[1:]
    use_nemo = "--dummy" not in args
    wav_path = "test.wav"
    output_path = "transcript.txt"
    if "--file" in args:
        idx = args.index("--file")
        if idx + 1 < len(args):
            wav_path = args[idx + 1]
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_path = args[idx + 1]
    main(use_nemo=use_nemo, wav_path=wav_path, output_path=output_path)
