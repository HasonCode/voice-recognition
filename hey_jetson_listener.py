"""
Listen for "hey jetson" or "hey jet" (fuzzy + regex) in the live transcript, then capture
everything said after it until there is a pause in speech. The response is
printed, optionally saved, and read aloud with AI voice (TTS).

Uses the same live pipeline as pipeline_live_test; the wake-word logic runs
on the streamed transcript. Captured questions are answered by a local GGUF
Llama model (if found), then spoken with TTS.

Usage:
  python hey_jetson_listener.py
  python hey_jetson_listener.py --list-devices   # pick mic index for -d
  python hey_jetson_listener.py --output responses.txt
  python hey_jetson_listener.py --similarity 0.82     # stricter wake fuzzy match (0–1)
  python hey_jetson_listener.py --no-voice              # disable TTS
  python hey_jetson_listener.py --voice-id en-US-GuyNeural
  export APPLIO_RVC_CMD='conda run -n applio-rvc python -m rvc_python cli --input {input_wav} --output {output_wav} --model {model_path} --index {index_path} --device cpu:0'
  pip install edge-tts   # for AI-style voice (needs internet to generate)
  pip install pyttsx3    # offline fallback (e.g. espeak on Linux)
  export DEEPGRAM_API_KEY=...   # for cloud ASR (--deepgram)
  poetry install -E deepgram   # or: pip install deepgram-sdk
"""

import os
import sys
import re
import threading
import time
import math
import subprocess
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from voice_recognition.audio.config import AudioConfig
from voice_recognition.decoder import CTCPrefixBeamSearch
from voice_recognition.pipeline import (
    DeepgramStreamingPipeline,
    StreamingCaptionPipeline,
    StreamingConfig,
)
from voice_recognition.postprocess import merge_display_into_transcript
from voice_recognition.stabilizer import CaptionStabilizer
from voice_recognition.models import LazyLlamaGguf, find_gguf_in_dir
from voice_recognition.wakeword import (
    DEFAULT_DONE_PHRASES,
    DEFAULT_SIMILARITY_THRESHOLD,
    HeyJetsonListener,
    play_audio_file,
    speak,
)

NEMO_MODELS_DIR = Path(__file__).resolve().parent / "src" / "voice_recognition" / "models"
NEMO_MODEL_SMALL = NEMO_MODELS_DIR / "ctc_small.nemo"
NEMO_MODEL_LARGE = NEMO_MODELS_DIR / "stt_en_conformer_ctc_large.nemo"
LLM_MODELS_DIR = Path(__file__).resolve().parent / "src" / "voice_recognition" / "models"
VOCAB_DUMMY = ["<blank>", "a", "b", "c", " "]
BLANK_DUMMY = 0
DEFAULT_RVC_MODEL = Path(__file__).resolve().parent / "hason_voice_200e_22800s.pth"
DEFAULT_RVC_INDEX = Path(__file__).resolve().parent / "hason_voice.index"

# Spoken after the wake phrase is detected (before the user’s follow-up question).
WAKE_PROMPT = "What is your request?"
# Short sting played (with ASR paused) immediately before ``WAKE_PROMPT`` TTS.
QUESTION_INTRO_WAV = (
    Path(__file__).resolve().parent / "Fade-In - Sound Effect - Trivia King (128k).wav"
)
THINKING_WAV = Path(__file__).resolve().parent / "thinking.wav"
LLM_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in one short, complete sentence unless the user asks for detail. "
    "Be accurate and concise. Do not invent what the user asked. "
    "If the request is unclear, ask one short clarification."
)
LLM_MAX_TOKENS = 160
LLM_N_CTX = 512
LLM_N_BATCH = 32


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
    pause_sec=1.3,
    similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    voice=True,
    voice_id="en-US-AriaNeural",
    llm_model_path=None,
    done_phrases=DEFAULT_DONE_PHRASES,
    nemo_model_path=None,
    rvc_model_path=None,
    rvc_index_path=None,
    backend: str = "nemo",
    deepgram_model: str = "nova-3",
    deepgram_api_key: str | None = None,
    wait_music_path: str | None = None,
):
    config = StreamingConfig(context_sec=1.6, update_interval_sec=0.25)

    model_forward = None
    decoder = None
    model_input = "mel"
    model_name = "dummy"

    if backend == "deepgram":
        dg_key = (deepgram_api_key or os.environ.get("DEEPGRAM_API_KEY") or "").strip()
        if not dg_key:
            print(
                "Deepgram backend requires DEEPGRAM_API_KEY (or pass --deepgram-api-key).",
                file=sys.stderr,
            )
            sys.exit(1)
        model_name = f"Deepgram ({deepgram_model})"
    else:
        if nemo_model_path:
            nemo_path = Path(nemo_model_path)
            if not nemo_path.is_absolute():
                nemo_path = NEMO_MODELS_DIR / nemo_path
        else:
            # Prefer the larger model for better recognition accuracy.
            nemo_path = NEMO_MODEL_LARGE if NEMO_MODEL_LARGE.exists() else NEMO_MODEL_SMALL

        if use_nemo and nemo_path.exists():
            from voice_recognition.models import load_nemo_model

            print(f"Loading NeMo model from {nemo_path}...")
            model_forward, vocab, blank_index = load_nemo_model(nemo_path)
            decoder = CTCPrefixBeamSearch(vocab, blank_index=blank_index, beam_size=8)
            model_input = "audio"
            model_name = "NeMo"
        else:
            if use_nemo:
                print(f"NeMo model not found at {nemo_path}, using dummy model.")
            model_forward = make_dummy_model()
            decoder = CTCPrefixBeamSearch(VOCAB_DUMMY, blank_index=BLANK_DUMMY, beam_size=8)
            model_input = "mel"
            model_name = "dummy"

    accumulated_transcript = [""]
    last_display = [None]
    identical_count = [0]
    last_question_sig = [""]
    last_question_ts = [0.0]
    last_answered_sig = [""]
    last_answered_ts = [0.0]
    recent_answered_sigs: list[tuple[str, float]] = []
    POST_ANSWER_SUPPRESS_SEC = 12.0
    STUCK_THRESHOLD = 8
    pipeline_ref = [None]
    listener_ref: list = [None]

    llm_path = Path(llm_model_path) if llm_model_path else find_gguf_in_dir(LLM_MODELS_DIR)
    if llm_path is None:
        print(f"No GGUF model found in {LLM_MODELS_DIR}. Responses will echo transcript only.")
        llm = None
    else:
        print(f"Using GGUF LLM: {llm_path}")
        llm = LazyLlamaGguf(llm_path, n_ctx=LLM_N_CTX, n_gpu_layers=0, n_batch=LLM_N_BATCH)
    wait_music = Path(wait_music_path).expanduser() if wait_music_path else THINKING_WAV
    if wait_music_path and not wait_music.is_absolute():
        wait_music = Path(__file__).resolve().parent / wait_music

    # Refcount > 0: streaming loop reads mic but skips ASR while LLM/TTS is active.
    hold_lock = threading.Lock()
    hold_n = [0]
    turn_lock = threading.Lock()
    turn_state = ["idle"]  # idle -> prompting -> listening -> answering

    def asr_gate_ok() -> bool:
        with hold_lock:
            return hold_n[0] == 0

    def get_turn_state() -> str:
        with turn_lock:
            return turn_state[0]

    def can_start_question_turn() -> bool:
        return get_turn_state() == "idle"

    def set_turn_state(state: str) -> None:
        with turn_lock:
            turn_state[0] = state

    def try_begin_prompting() -> bool:
        with turn_lock:
            if turn_state[0] != "idle":
                return False
            turn_state[0] = "prompting"
            return True

    def finish_question_turn() -> None:
        set_turn_state("idle")

    def run_with_asr_paused(job, *, on_paused: Optional[Callable[[], None]] = None) -> None:
        """Run a background job while ASR is paused (LLM/TTS).

        ``on_paused`` runs on this thread immediately after the pause begins (before ``job``).
        """
        with hold_lock:
            hold_n[0] += 1
        if on_paused is not None:
            try:
                on_paused()
            except Exception:
                with hold_lock:
                    hold_n[0] -= 1
                raise

        def runner() -> None:
            try:
                job()
            finally:
                with hold_lock:
                    hold_n[0] -= 1

        threading.Thread(target=runner, daemon=True).start()

    def speak_pausing_asr(text: str) -> None:
        """Run TTS while ASR is paused so playback is not transcribed."""
        t = text.strip()
        if not t:
            return
        run_with_asr_paused(
            lambda: speak(
                t,
                voice=voice_id,
                use_thread=False,
                rvc_model_path=rvc_model_path,
                rvc_index_path=rvc_index_path,
            )
        )

    def answer_question(question: str) -> str:
        q = normalize_request_text(question)
        if not q:
            return ""
        if llm is not None:
            try:
                raw = llm.generate_short_reply(
                    q,
                    system=LLM_SYSTEM_PROMPT,
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=0.15,
                )
                return shorten_answer(raw)
            except Exception as e:
                return f"I hit an error while generating an answer: {e}"
        if looks_garbled(q):
            return "I could not parse that clearly. Please repeat your question in one short sentence."
        fast = try_fast_math(q)
        if fast is not None:
            return fast
        return q

    def start_thinking_loop() -> threading.Event:
        stop_event = threading.Event()
        if not wait_music.is_file():
            return stop_event

        def play_once_interruptible() -> bool:
            path = str(wait_music)
            commands = [
                ["mpv", "--no-video", "--really-quiet", path],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
            ]
            if path.lower().endswith(".wav"):
                commands.append(["aplay", "-q", path])
            for cmd in commands:
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except FileNotFoundError:
                    continue
                while proc.poll() is None:
                    if stop_event.wait(0.05):
                        proc.terminate()
                        try:
                            proc.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return True
                if proc.returncode == 0:
                    return True
            return False

        def loop() -> None:
            while not stop_event.is_set():
                if not play_once_interruptible():
                    return

        threading.Thread(target=loop, daemon=True).start()
        return stop_event

    def try_fast_math(text: str) -> str | None:
        """
        Fast deterministic arithmetic for simple spoken math to avoid LLM overhead.
        Handles patterns like 'what is five plus five' and '12 times 4'.
        """
        words_to_num = {
            "zero": 0.0,
            "one": 1.0,
            "two": 2.0,
            "three": 3.0,
            "four": 4.0,
            "five": 5.0,
            "six": 6.0,
            "seven": 7.0,
            "eight": 8.0,
            "nine": 9.0,
            "ten": 10.0,
            "eleven": 11.0,
            "twelve": 12.0,
            "thirteen": 13.0,
            "fourteen": 14.0,
            "fifteen": 15.0,
            "sixteen": 16.0,
            "seventeen": 17.0,
            "eighteen": 18.0,
            "nineteen": 19.0,
            "twenty": 20.0,
        }
        cleaned = re.sub(r"\b(what is|what's|calculate|compute)\b", " ", text.lower())
        cleaned = re.sub(r"[?!]+$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        patterns = [
            (r"^([a-z0-9.\-]+)\s+(plus)\s+([a-z0-9.\-]+)$", "+"),
            (r"^([a-z0-9.\-]+)\s+(minus)\s+([a-z0-9.\-]+)$", "-"),
            (r"^([a-z0-9.\-]+)\s+(times|multiplied by)\s+([a-z0-9.\-]+)$", "*"),
            (r"^([a-z0-9.\-]+)\s+(divided by|over)\s+([a-z0-9.\-]+)$", "/"),
        ]

        def parse_num(tok: str) -> float | None:
            tok = tok.strip()
            if tok in words_to_num:
                return words_to_num[tok]
            try:
                return float(tok)
            except ValueError:
                return None

        for pat, op in patterns:
            m = re.match(pat, cleaned)
            if not m:
                continue
            a = parse_num(m.group(1))
            b = parse_num(m.group(3))
            if a is None or b is None:
                continue
            if op == "+":
                val = a + b
            elif op == "-":
                val = a - b
            elif op == "*":
                val = a * b
            else:
                if math.isclose(b, 0.0):
                    return "Division by zero is undefined."
                val = a / b
            if float(val).is_integer():
                return str(int(val))
            return f"{val:.4f}".rstrip("0").rstrip(".")
        return None

    def shorten_answer(text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
        """Keep replies short and sane for low-latency TTS."""
        t = " ".join(text.strip().split())
        if not t:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", t)
        picked: list[str] = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            picked.append(p)
            if len(picked) >= max_sentences:
                break
        if picked and picked[-1][-1:] in ".!?":
            out = " ".join(picked)
        else:
            complete = [p for p in picked if p and p[-1:] in ".!?"]
            out = " ".join(complete) if complete else (picked[0] if picked else t)
        if len(out) > max_chars:
            out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:- ")
            if out and out[-1] not in ".!?":
                out += "."
        elif out and out[-1] not in ".!?":
            # The model likely hit the token cap mid-sentence; avoid reading a dangling fragment.
            clipped = re.split(r"(?<=[.!?])\s+", out)
            complete = [p for p in clipped if p and p[-1:] in ".!?"]
            if complete:
                out = " ".join(complete)
            else:
                out = out.rsplit(" ", 1)[0].rstrip(",;:- ") + "."
        return out

    def normalize_request_text(text: str) -> str:
        """Cleanup for common streaming ASR split/glue artifacts before LLM."""
        t = normalize_stream_text(text)
        if not t:
            return ""
        fixes = (
            (r"\bwhat'\b", "what is"),
            (r"\bwh\s+at\b", "what"),
            (r"\bwh\s+ere\b", "where"),
            (r"\bwh\s+en\b", "when"),
            (r"\bwh\s+y\b", "why"),
            (r"\bho\s+w\b", "how"),
            (r"\bca\s+n\b", "can"),
            (r"\bdo\s+es\b", "does"),
            (r"\bis\s+it\b", "is it"),
            (r"\bte\s+ll\b", "tell"),
            (r"\bex\s+plain\b", "explain"),
        )
        out = t.lower()
        for pat, rep in fixes:
            out = re.sub(pat, rep, out)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def question_signature(text: str) -> str:
        """Stable key for suppressing stale repeated captions after an answer."""
        t = normalize_request_text(text)
        if not t:
            return ""
        t = re.sub(r"\bhey,?\s*(?:jetson|jet\s*son|jet)\b", " ", t)
        t = re.sub(r"\bjets?\s+on\b", " ", t)
        t = re.sub(r"\b(?:jet\s*son|jetson|jet|jets)\s+(?:done|over)\b.*$", " ", t)
        t = re.sub(r"[^a-z0-9]+", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def is_recent_answered_caption(text: str, now: float | None = None) -> bool:
        sig = question_signature(text)
        if not sig:
            return False
        now_ts = time.time() if now is None else now
        recent_answered_sigs[:] = [
            (answered_sig, answered_ts)
            for answered_sig, answered_ts in recent_answered_sigs
            if now_ts - answered_ts <= POST_ANSWER_SUPPRESS_SEC
        ]
        if last_answered_sig[0] and all(s != last_answered_sig[0] for s, _ in recent_answered_sigs):
            recent_answered_sigs.append((last_answered_sig[0], last_answered_ts[0]))
        if not recent_answered_sigs:
            return False
        for answered, _ in recent_answered_sigs:
            if not answered:
                continue
            if sig == answered or answered in sig or sig in answered:
                return True
        return False

    def normalize_stream_text(text: str) -> str:
        """General transcript normalization used for both wake parsing and LLM input."""
        out = " ".join(text.strip().split()).lower()
        if not out:
            return ""
        # Merge common split function words/question prefixes.
        merge_fixes = (
            (r"\bw\s+h\s+a\s+t\b", "what"),
            (r"\bw\s+h\s+e\s+r\s+e\b", "where"),
            (r"\bw\s+h\s+e\s+n\b", "when"),
            (r"\bw\s+h\s+y\b", "why"),
            (r"\bh\s+o\s+w\b", "how"),
            (r"\bwh\s+at\b", "what"),
            (r"\bwh\s+ere\b", "where"),
            (r"\bwh\s+en\b", "when"),
            (r"\bwh\s+y\b", "why"),
            (r"\bho\s+w\b", "how"),
            (r"\bdo\s+you\b", "do you"),
            (r"\bis\s+f\b", "is "),
            (r"\bwithh\b", "with"),
            (r"\bth\s+is\b", "this"),
            (r"\bfa\s+st\b", "fast"),
            (r"\bwi\s+th\b", "with"),
        )
        for pat, rep in merge_fixes:
            out = re.sub(pat, rep, out)
        # Remove obvious single-letter stutter fragments except useful pronouns.
        out = re.sub(r"\b(?!a\b|i\b)[a-z]\b", " ", out)
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def looks_garbled(text: str) -> bool:
        words = [w for w in re.findall(r"[a-z']+", text.lower()) if w]
        if not words:
            return True
        long_words = [w for w in words if len(w) >= 3]
        vowel_words = [w for w in words if re.search(r"[aeiou]", w)]
        if len(long_words) == 0:
            return True
        if len(vowel_words) / max(1, len(words)) < 0.5:
            return True
        # Highly repetitive gibberish often collapses to very low unique ratio.
        if len(set(words)) / max(1, len(words)) < 0.35 and len(words) >= 6:
            return True
        return False

    def on_response(response: str):
        """Called when user said 'hey jetson' and then spoke until a pause."""
        normalized = normalize_request_text(response)
        if not normalized:
            finish_question_turn()
            return
        normalized_sig = question_signature(normalized)
        now = time.time()
        if normalized_sig and normalized_sig == last_question_sig[0] and now - last_question_ts[0] < POST_ANSWER_SUPPRESS_SEC:
            finish_question_turn()
            return
        if is_recent_answered_caption(normalized, now):
            finish_question_turn()
            return
        set_turn_state("answering")
        last_question_sig[0] = normalized_sig or normalized
        last_question_ts[0] = now
        if listener_ref[0] is not None:
            listener_ref[0].flush_transcript_stream_after_llm_turn()
        # Start a fresh turn after each captured question to avoid stale text
        # making subsequent wake detection sluggish.
        accumulated_transcript[0] = ""
        last_display[0] = None
        identical_count[0] = 0
        print("\n[Question]", normalized)

        def answer_and_respond() -> None:
            thinking_stop = None
            try:
                print("[Generating] Thinking...")
                thinking_stop = start_thinking_loop()
                answer = answer_question(normalized)
                if thinking_stop is not None:
                    thinking_stop.set()
                if answer:
                    print("[Answer]", answer)
                if output_path:
                    path = Path(output_path)
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(f"Q: {normalized}\n")
                        if answer:
                            f.write(f"A: {answer}\n")
                        f.write("\n")
                    print(f"  (appended to {path})")
                if voice and answer:
                    speak(
                        answer,
                        voice=voice_id,
                        use_thread=False,
                        rvc_model_path=rvc_model_path,
                        rvc_index_path=rvc_index_path,
                        speed=1.4,
                    )
            finally:
                if thinking_stop is not None:
                    thinking_stop.set()
                accumulated_transcript[0] = ""
                last_display[0] = None
                identical_count[0] = 0
                if listener_ref[0] is not None:
                    listener_ref[0].flush_transcript_stream_after_llm_turn()
                last_answered_sig[0] = normalized_sig or question_signature(normalized)
                last_answered_ts[0] = time.time()
                if last_answered_sig[0]:
                    recent_answered_sigs.append((last_answered_sig[0], last_answered_ts[0]))
                    recent_answered_sigs[:] = recent_answered_sigs[-8:]
                last_question_ts[0] = last_answered_ts[0]
                finish_question_turn()

        run_with_asr_paused(answer_and_respond)

    def on_wake():
        if not try_begin_prompting():
            return

        def intro_then_prompt() -> None:
            try:
                if QUESTION_INTRO_WAV.is_file():
                    ok = play_audio_file(QUESTION_INTRO_WAV)
                    if not ok:
                        print(f"  (could not play intro sound: {QUESTION_INTRO_WAV})")
                if voice:
                    speak(
                        WAKE_PROMPT,
                        voice=voice_id,
                        use_thread=False,
                        rvc_model_path=rvc_model_path,
                        rvc_index_path=rvc_index_path,
                    )
            finally:
                if listener_ref[0] is not None:
                    listener_ref[0].resume_capture_timeout()
                set_turn_state("listening")

        def flush_caption_state() -> None:
            print(f"\n[Awaiting question] {WAKE_PROMPT}")
            accumulated_transcript[0] = ""
            last_display[0] = None
            identical_count[0] = 0
            pl = pipeline_ref[0]
            if pl is not None:
                stab = getattr(pl, "stabilizer", None)
                if stab is not None:
                    stab.reset()
            if listener_ref[0] is not None:
                listener_ref[0].pause_capture_timeout()
                listener_ref[0].flush_transcript_stream_after_llm_turn()

        run_with_asr_paused(intro_then_prompt, on_paused=flush_caption_state)

    listener = HeyJetsonListener(
        on_response=on_response,
        on_wake=on_wake,
        can_wake=can_start_question_turn,
        on_empty_capture=finish_question_turn,
        similarity_threshold=similarity_threshold,
        pause_sec=pause_sec,
        done_phrases=tuple(done_phrases),
    )
    listener_ref[0] = listener

    def on_display(s):
        if not asr_gate_ok():
            return
        s_norm = normalize_stream_text(s)
        if not listener.is_capturing() and is_recent_answered_caption(s_norm):
            accumulated_transcript[0] = ""
            last_display[0] = None
            identical_count[0] = 0
            if listener_ref[0] is not None:
                listener_ref[0].flush_transcript_stream_after_llm_turn()
            return
        capturing = listener.is_capturing()
        wake_module = None
        if capturing:
            from voice_recognition.wakeword import hey_jetson as wake_module

            if "?" in s_norm:
                current_question = accumulated_transcript[0]
                if wake_module._matches_trigger(current_question, similarity_threshold):
                    current_question = ""
                caption_question = wake_module._text_after_trigger(s_norm, similarity_threshold) or s_norm
                short_question_prefixes = {
                    "what is",
                    "what are",
                    "who is",
                    "who are",
                    "where is",
                    "where are",
                    "when is",
                    "why is",
                    "how is",
                    "how are",
                    "how many",
                    "how much",
                }
                if current_question in short_question_prefixes and not caption_question.startswith(current_question):
                    merged_question = f"{current_question} {caption_question}"
                else:
                    merged_question = merge_display_into_transcript(current_question, caption_question)
                accumulated_transcript[0] = ""
                last_display[0] = None
                identical_count[0] = 0
                print("display:", repr(s_norm))
                listener.push_transcript(merged_question)
                return

            caption_done_text, caption_done_hit = listener._pop_done_phrase(s_norm)
            if caption_done_hit:
                current_question = accumulated_transcript[0]
                if caption_done_text:
                    current_question = merge_display_into_transcript(current_question, caption_done_text)
                merged_done = " ".join((current_question + " " + s_norm).split()).strip()
                accumulated_transcript[0] = ""
                last_display[0] = None
                identical_count[0] = 0
                print("display:", repr(s_norm))
                listener.push_transcript(merged_done)
                return
        if s_norm == last_display[0]:
            identical_count[0] += 1
            if identical_count[0] == STUCK_THRESHOLD and pipeline_ref[0] is not None:
                stab = getattr(pipeline_ref[0], "stabilizer", None)
                if stab is not None:
                    stab.reset()
            return
        identical_count[0] = 0
        last_display[0] = s_norm
        transcript_base = accumulated_transcript[0]
        if capturing:
            if wake_module._matches_trigger(transcript_base, similarity_threshold):
                transcript_base = ""
        merged = merge_display_into_transcript(transcript_base, s_norm)
        if capturing:
            if wake_module is None:
                from voice_recognition.wakeword import hey_jetson as wake_module
            tail = wake_module._text_after_trigger(merged, similarity_threshold)
            if tail.strip():
                merged = tail
            elif wake_module._matches_trigger(merged, similarity_threshold):
                accumulated_transcript[0] = ""
                print("display:", repr(s_norm))
                listener.push_transcript(merged)
                return
        merged = " ".join(merged.split()).strip()
        accumulated_transcript[0] = merged
        print("display:", repr(s_norm))
        if not merged:
            return
        was_capturing = capturing
        listener.push_transcript(merged)
        if not was_capturing and listener.is_capturing():
            from voice_recognition.wakeword import hey_jetson as wake_module

            tail = wake_module._text_after_trigger(merged, similarity_threshold)
            accumulated_transcript[0] = tail.strip() if tail.strip() else ""

    if backend == "deepgram":
        pipeline = DeepgramStreamingPipeline(
            audio_config=AudioConfig(),
            chunk_duration_sec=config.update_interval_sec,
            on_display=on_display,
            asr_enabled=asr_gate_ok,
            model=deepgram_model,
            api_key=deepgram_api_key,
        )
    else:
        pipeline = StreamingCaptionPipeline(
            config=config,
            audio_config=AudioConfig(),
            model_forward=model_forward,
            model_input=model_input,
            decoder=decoder,
            stabilizer=CaptionStabilizer(stable_n=2),
            on_display=on_display,
            asr_enabled=asr_gate_ok,
        )
    pipeline_ref[0] = pipeline

    print(
        f"Running with {model_name} (wake similarity={similarity_threshold}). "
        "Say 'Hey Jetson' or 'Hey Jet', then your question; end with 'jet done', 'jetson done', or pause."
    )
    if backend == "deepgram":
        print("Transcription uses Deepgram's cloud API (requires outbound HTTPS/WebSocket).")
    if voice:
        print("AI voice will read back each captured response.")
        print(f"After the wake phrase, it will say: {WAKE_PROMPT}")
    else:
        print(f"After the wake phrase, it will print (no TTS): {WAKE_PROMPT}")
    if output_path:
        print(f"Responses will be appended to {output_path}\n")
    try:
        pipeline.run(device=device)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if hasattr(pipeline, "stop"):
            pipeline.stop()
        listener.stop()
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Listen for 'Hey Jetson' / 'Hey Jet', capture speech until pause or done phrase.")
    p.add_argument("--output", "-o", type=str, help="Append each response to this file")
    p.add_argument(
        "--device",
        "-d",
        type=int,
        default=None,
        help="Microphone device index (same numbering as --list-devices)",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="Print sounddevice audio devices and exit (use index with -d)",
    )
    p.add_argument("--pause", "-p", type=float, default=1.3, help="Seconds of silence to end capture")
    p.add_argument(
        "--done-phrase",
        action="append",
        default=[],
        help="Phrase that immediately ends question capture when spoken (repeatable).",
    )
    p.add_argument(
        "--similarity",
        "-s",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        metavar="X",
        help=(
            "Fuzzy / letter-similarity floor for matching the wake phrase (0–1). "
            f"Default {DEFAULT_SIMILARITY_THRESHOLD} (good for ASR like 'hejets on'); "
            "raise toward 1.0 for fewer false triggers."
        ),
    )
    p.add_argument("--dummy", action="store_true", help="Use dummy model (ignored when --deepgram is set)")
    p.add_argument(
        "--deepgram",
        action="store_true",
        help="Stream mic audio to Deepgram for transcription (set DEEPGRAM_API_KEY or use --deepgram-api-key)",
    )
    p.add_argument(
        "--deepgram-model",
        type=str,
        default="nova-3",
        metavar="MODEL",
        help="Deepgram model for live Listen v1 (default: nova-3)",
    )
    p.add_argument(
        "--deepgram-api-key",
        type=str,
        default=None,
        help="Deepgram API key for this run (default: environment DEEPGRAM_API_KEY)",
    )
    p.add_argument(
        "--nemo-model",
        type=str,
        default=None,
        help="Path (or filename in src/voice_recognition/models) to a .nemo checkpoint. Defaults to large if present.",
    )
    p.add_argument("--llm-model", type=str, default=None, help="Path to GGUF model (defaults to first *.gguf in models dir)")
    p.add_argument("--no-voice", action="store_true", help="Disable TTS reading of responses")
    p.add_argument("--voice-id", type=str, default="en-US-AriaNeural", help="Edge TTS voice (e.g. en-US-GuyNeural)")
    p.add_argument(
        "--rvc-model",
        type=str,
        default=None,
        help="Optional path to Applio/RVC .pth voice model used for TTS conversion.",
    )
    p.add_argument(
        "--rvc-index",
        type=str,
        default=None,
        help="Optional path to Applio/RVC .index file used for TTS conversion.",
    )
    p.add_argument(
        "--wait-music",
        type=str,
        default=None,
        help="Audio file to loop while waiting for the LLM response (default: thinking.wav).",
    )
    args = p.parse_args()
    if args.list_devices:
        try:
            import sounddevice as sd

            print(sd.query_devices())
            print('\nUse the left-hand "index" of your microphone with: python hey_jetson_listener.py -d N')
        except ImportError:
            print("sounddevice not installed: pip install sounddevice", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    main(
        use_nemo=not args.dummy and not args.deepgram,
        device=args.device,
        output_path=args.output,
        pause_sec=args.pause,
        similarity_threshold=args.similarity,
        voice=not args.no_voice,
        voice_id=args.voice_id,
        llm_model_path=args.llm_model,
        done_phrases=tuple(args.done_phrase) if args.done_phrase else DEFAULT_DONE_PHRASES,
        nemo_model_path=args.nemo_model,
        rvc_model_path=args.rvc_model,
        rvc_index_path=args.rvc_index,
        backend="deepgram" if args.deepgram else "nemo",
        deepgram_model=args.deepgram_model,
        deepgram_api_key=args.deepgram_api_key,
        wait_music_path=args.wait_music,
    )
