"""
Listen for "hey jetson" or short "hey jet" (with fuzzy + regex match), then capture
the following speech until a pause or an end phrase (e.g. "jet done", "jetson done")
and feed it to a callback.

Uses transcript stream: when the end of the transcript matches the trigger phrase,
we enter capture mode and accumulate text until no new speech for pause_sec.
"""

import re
import threading
import time
from difflib import SequenceMatcher
from typing import Callable, Optional

# Default trigger phrase (normalized: lowercase, single spaces)
TRIGGER_PHRASE = "hey jetson"
# Letters-only canonical form of the wake phrase (spaces removed)
_WAKE_LETTERS = "heyjetson"

# Regex: "hey" then optional space then "jetson" at end of text
TRIGGER_REGEX = re.compile(
    r"\bhey,?\s*jetson\s*$",
    re.IGNORECASE,
)

# ASR often splits/glues differently than "hey jetson" — match common variants at **end** of transcript.
_WAKE_AT_END = re.compile(
    r"(?:"
    r"\bhey,?\s*jetson"
    r"|\bhay\s*jetson"
    r"|\bhayjetson"
    r"|\bjetson"
    r"|\bhey,?\s+jets?\s+on"
    r"|\bheyjets?\s+on"
    r"|\bh\s+ay\s+jets?\s+on"
    r"|\bhe+jets?\s+on"
    r"|\bhey,?\s+jet\s*son"
    r"|\bhey,?\s+jetds\s+on"
    r")\s*$",
    re.IGNORECASE,
)

# Short wake: "hey jet" at end (does not match "hey jetson" — more text after "jet").
_HEY_JET_AT_END = re.compile(r"\bhey\s+jet\s*$", re.IGNORECASE)

# Default fuzzy similarity (0–1) for last-word match vs "hey jetson".
# ~0.70 is lenient enough for noisy ASR (e.g. "he jets on"); raise via CLI for fewer false triggers.
DEFAULT_SIMILARITY_THRESHOLD = 0.70

# Seconds of no new transcript to consider "pause in speech"
DEFAULT_PAUSE_SEC = 2.5
# Longer phrases first are not required here: ``_pop_done_phrase`` sorts by length.
# "jetson done" must beat plain "done" so questions like "... france jetson done" do not
# leave a stray "jetson" on the question text.
# Plain "done"/"over" are intentionally omitted: phrases like "is that done?" would
# false-trigger. Use jet/jetson-specific closers or add ``--done-phrase done`` if needed.
DEFAULT_DONE_PHRASES = (
    "jets on done",
    "jetson done",
    "jet son done",
    "jet done",
    "that is all",
    "thank you",
)
# Tolerant end-of-question closers on punctuation-loosened text (Deepgram often adds
# commas/periods: "jet, done.", "jet done.").
_DONE_TAIL_REGEX = re.compile(
    r"(?:"
    r"\b(?:jet\s*son|jets?\s*on|jetsen|jetson|jettson)\s*[.,]?\s*"
    r"(?:done|over|that\s+is\s+all|thank\s+you)\b"
    r"|"
    r"\bjet\s+son\s*[.,]?\s*(?:done|over|that\s+is\s+all|thank\s+you)\b"
    r"|"
    r"\b(?:jet|jets)\s*[.,]?\s*(?:done|over|that\s+is\s+all|thank\s+you)\b"
    r"|"
    r"\b(?:that\s+is\s+all|thank\s+you)\b"
    r")[\s.!?]*$",
    re.IGNORECASE,
)

# "is that done?" must not count as an end phrase.
_FALSE_THAT_DONE_TAIL = re.compile(r"\bthat\s+done\b[\s.!?]*$", re.IGNORECASE)

# ASR mis-hears that fuzzy-match "hey jet" / tail scores at low thresholds.
_FALSE_WAKE_ASR_NAMES = re.compile(
    r"\bhey,?\s*(?:jack|jen|jeff|jenn|jensen|jackson|jacks|johnson)\b",
    re.IGNORECASE,
)

# How often to check for pause (sec)
PAUSE_CHECK_INTERVAL = 0.3

# Cloud ASR (e.g. Deepgram) may resend the same final line many times. Without this,
# each identical "hey jetson" tail re-triggers ``on_wake`` after capture times out empty.
DEFAULT_WAKE_PROMPT_DEBOUNCE_SEC = 3.5
DEFAULT_EMPTY_CAPTURE_GRACE_SEC = 8.0


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip()


def _loose_done_match_text(text: str) -> str:
    """Lowercase, collapse punctuation to spaces, trim trailing sentence punct (for done matching)."""
    t = text.lower()
    t = re.sub(r"[,;:\"'`]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[\s.!?]+$", "", t)
    return t.strip()


def _debounce_wake_key(norm: str) -> str:
    """Punctuation-invariant key so 'hey, jetson.' and 'hey jetson' debounce as the same wake."""
    w = _letters_only(norm)
    return w[-18:] if len(w) > 18 else w


def _last_n_words(text: str, n: int) -> str:
    """Last n words of text, normalized."""
    words = text.split()
    return " ".join(words[-n:]) if words else ""


def _similarity(a: str, b: str) -> float:
    """Sequence similarity in [0, 1]."""
    a, b = _normalize(a), _normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _letters_only(text: str) -> str:
    """Lowercase letters only (ASR glue/spacing invariant)."""
    return re.sub(r"[^a-z]+", "", text.lower())


def _letter_tail_wake_score(letters: str) -> float:
    """How closely the end of `letters` matches ``heyjetson`` (0–1)."""
    if not letters:
        return 0.0
    canon = _WAKE_LETTERS
    best = 0.0
    # Compare trailing slices; ASR may prefix garbage on the same tail window.
    max_take = min(len(letters), len(canon) + 6)
    for take in range(7, max_take + 1):
        frag = letters[-take:]
        best = max(best, SequenceMatcher(None, frag, canon).ratio())
    return best


def _letter_wake_threshold(word_threshold: float) -> float:
    """Slightly more lenient threshold for letters-only wake (BPE spacing)."""
    return min(0.94, max(0.73, word_threshold + 0.12))


def _letters_contain_jet_anchor(letters: str) -> bool:
    """Require consecutive ``jet`` so names like ``heyjen`` / ``heyjeff`` do not fuzzy-match."""
    return "jet" in letters


def _matches_trigger(text: str, threshold: float) -> bool:
    """True if end of text matches 'hey jetson' or short 'hey jet' via regex or fuzzy match."""
    if not text or not text.strip():
        return False
    norm = _normalize(text)
    if _FALSE_WAKE_ASR_NAMES.search(norm):
        return False
    norm = norm.strip(" \t\r\n.,!?")
    # Regex: canonical or ASR-variant wake at end of string
    if TRIGGER_REGEX.search(norm) or _WAKE_AT_END.search(norm) or _HEY_JET_AT_END.search(norm):
        return True
    # Fuzzy: compare last 2–5 words to trigger phrase (word-level); require ``jet`` in letters
    # so low-threshold similarity does not fire on "hey, jeff." / "hey, jen." etc.
    for n in (2, 3, 4, 5):
        suffix = _last_n_words(norm, n)
        if not suffix:
            continue
        lt_suf = _letters_only(suffix)
        if suffix == "hey jet":
            return True
        if _similarity(suffix, "hey jet") >= min(0.88, threshold + 0.1):
            if _letters_contain_jet_anchor(lt_suf):
                return True
        if suffix.endswith(TRIGGER_PHRASE) or _similarity(suffix, TRIGGER_PHRASE) >= threshold:
            if _letters_contain_jet_anchor(lt_suf):
                return True
    # Letters-only tail: "heyjets on", "h ayjets on", "hejets on" → heyjetson
    tail_words = _last_n_words(norm, 6)
    lt = _letters_only(tail_words)
    if _letters_contain_jet_anchor(lt) and _letter_tail_wake_score(lt) >= _letter_wake_threshold(threshold):
        return True
    return False


def _strip_trailing_wake_words(norm: str, threshold: float) -> Optional[str]:
    """If the last 1–7 words are a noisy wake, return text before them; else None.

    Longest suffix first so we do not strip a short false positive (e.g. ``ayjets on``)
    and leave a stray ``h`` from ``h ayjets on``.
    """
    words = norm.split()
    if not words:
        return None
    for n in range(min(8, len(words)), 0, -1):
        tail = " ".join(words[-n:])
        if _matches_trigger(tail, threshold):
            return " ".join(words[:-n]).strip()
    return None


def _text_after_trigger(text: str, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> str:
    """Return the part of text after the last occurrence of trigger-like phrase."""
    norm = _normalize(text)
    for pat in (
        re.compile(r"\bhey,?\s*jetson\b", re.IGNORECASE),
        re.compile(r"\bhay\s*jetson\b|\bhayjetson\b|\bjetson\b", re.IGNORECASE),
        re.compile(
            r"\bhey,?\s+jets?\s+on\b|\bheyjets?\s+on\b|\bh\s+ay\s+jets?\s+on\b|\bhe+jets?\s+on\b"
            r"|\bhey,?\s+jet\s*son\b|\bhey,?\s+jetds\s+on\b",
            re.IGNORECASE,
        ),
        # Short wake; do not match the "hey jet" prefix of "hey jet son" (common ASR for jetson).
        re.compile(r"\bhey,?\s+jet\b(?!\s*son\b)", re.IGNORECASE),
    ):
        last_end = -1
        for m in pat.finditer(norm):
            last_end = m.end()
        if last_end >= 0:
            return norm[last_end:].strip(" \t\r\n.,!")
    stripped = _strip_trailing_wake_words(norm, similarity_threshold)
    if stripped is not None:
        return stripped
    return ""


def question_text_for_llm(raw: str, similarity_threshold: float) -> str:
    """Strip embedded wake phrasing so only the user's question is sent to the LLM."""
    t = _normalize(raw)
    if not t:
        return ""
    tail = _text_after_trigger(t, similarity_threshold)
    return tail.strip() if tail.strip() else t


class HeyJetsonListener:
    """
    Listens to a stream of transcript updates. When "hey jetson" or "hey jet" is
    detected at the end of the transcript, captures all following speech until a pause
    (no new content for pause_sec), then calls on_response with that text.

    Optional ``on_wake`` is invoked once (outside the listener lock) immediately
    after the wake phrase is recognized, before the pause timer fires.
    """

    def __init__(
        self,
        on_response: Callable[[str], None],
        *,
        trigger_phrase: str = TRIGGER_PHRASE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        pause_sec: float = DEFAULT_PAUSE_SEC,
        done_phrases: tuple[str, ...] = DEFAULT_DONE_PHRASES,
        on_wake: Optional[Callable[[], None]] = None,
        can_wake: Optional[Callable[[], bool]] = None,
        on_empty_capture: Optional[Callable[[], None]] = None,
        wake_prompt_debounce_sec: float = DEFAULT_WAKE_PROMPT_DEBOUNCE_SEC,
        empty_capture_grace_sec: float = DEFAULT_EMPTY_CAPTURE_GRACE_SEC,
    ):
        self.on_response = on_response
        self.similarity_threshold = similarity_threshold
        self.pause_sec = pause_sec
        self.done_phrases = tuple(_normalize(x) for x in done_phrases if _normalize(x))
        self._on_wake = on_wake
        self._can_wake = can_wake
        self._on_empty_capture = on_empty_capture
        self._wake_prompt_debounce_sec = float(wake_prompt_debounce_sec)
        self._empty_capture_grace_sec = float(empty_capture_grace_sec)
        self._trigger_phrase = _normalize(trigger_phrase)

        self._last_transcript = ""
        self._last_update_time: float = 0.0
        self._capturing = False
        self._capture_timeout_paused = False
        self._response_parts: list[str] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._stop = False
        self._last_wake_debounce_key: str = ""
        self._last_on_wake_time: float = 0.0

    def is_capturing(self) -> bool:
        """True while listening for the question after a wake phrase."""
        with self._lock:
            return self._capturing

    def flush_transcript_stream_after_llm_turn(self) -> None:
        """Clear the caption diff baseline after a question was sent to the LLM."""
        with self._lock:
            self._last_transcript = ""

    def pause_capture_timeout(self) -> None:
        """Keep the current question capture open while prompt audio is playing."""
        with self._lock:
            self._capture_timeout_paused = True

    def resume_capture_timeout(self) -> None:
        """Resume question pause detection after prompt audio has finished."""
        with self._lock:
            self._capture_timeout_paused = False
            self._last_update_time = time.time()

    def _pop_done_phrase(self, text: str) -> tuple[str, bool]:
        """Strip a trailing done phrase from text. Returns (clean_text, matched)."""
        norm = _normalize(text)
        loose = _loose_done_match_text(norm)
        if not loose:
            return "", False
        m = _DONE_TAIL_REGEX.search(loose)
        if m is not None:
            trimmed = loose[: m.start()].strip()
            return trimmed, True
        for phrase in sorted(self.done_phrases, key=len, reverse=True):
            if not phrase:
                continue
            if phrase == "done" and _FALSE_THAT_DONE_TAIL.search(loose):
                continue
            if loose.endswith(phrase):
                trimmed = loose[: -len(phrase)].strip()
                return trimmed, True
        return loose, False

    def _schedule_pause_check(self) -> None:
        def check():
            if self._stop:
                return
            response_to_emit = ""
            empty_capture_callback: Optional[Callable[[], None]] = None
            with self._lock:
                if not self._capturing:
                    return
                if self._capture_timeout_paused:
                    self._last_update_time = time.time()
                elapsed = time.time() - self._last_update_time
                timeout_sec = self.pause_sec if self._response_parts else max(self.pause_sec, self._empty_capture_grace_sec)
                if elapsed >= timeout_sec:
                    # Timeout capture even when no words followed wake phrase; otherwise
                    # we can get stuck in capture mode forever after a false wake.
                    if self._response_parts:
                        response = " ".join(self._response_parts).strip()
                        response, _ = self._pop_done_phrase(response)
                        self._response_parts = []
                        self._capturing = False
                        response = question_text_for_llm(response, self.similarity_threshold)
                        if response:
                            response_to_emit = response
                        self._last_transcript = ""
                        self._capture_timeout_paused = False
                    self._capturing = False
                    self._response_parts = []
                    self._last_transcript = ""
                    self._capture_timeout_paused = False
                    if not response_to_emit:
                        empty_capture_callback = self._on_empty_capture
            if response_to_emit:
                try:
                    self.on_response(response_to_emit)
                except Exception:
                    pass
                return
            if empty_capture_callback is not None:
                try:
                    empty_capture_callback()
                except Exception:
                    pass
                return
            # Still capturing and not paused yet; check again
            self._timer = threading.Timer(PAUSE_CHECK_INTERVAL, check)
            self._timer.daemon = True
            self._timer.start()

        self._timer = threading.Timer(PAUSE_CHECK_INTERVAL, check)
        self._timer.daemon = True
        self._timer.start()

    def push_transcript(self, transcript: str) -> None:
        """
        Call this on every transcript update (e.g. from the pipeline display callback).
        """
        wake_callback: Optional[Callable[[], None]] = None
        response_callback_text = ""
        with self._lock:
            now = time.time()
            prev = self._last_transcript
            prev_update_time = self._last_update_time
            self._last_transcript = transcript
            self._last_update_time = now

            if self._capturing:
                wake_tail = _text_after_trigger(transcript, self.similarity_threshold)
                if _matches_trigger(transcript, self.similarity_threshold) and not wake_tail.strip():
                    # Deepgram can resend the same final wake after the prompt. While already
                    # collecting a question, a wake-only caption is stale context, not content.
                    self._last_transcript = ""
                    self._last_update_time = prev_update_time
                    return
                if "?" in transcript:
                    final_text = wake_tail if wake_tail.strip() else transcript
                    self._response_parts = [final_text]
                    self._last_transcript = final_text
                    self._last_update_time = now
                    merged = final_text.strip()
                    cleaned, done_hit = self._pop_done_phrase(merged)
                    question_mark_hit = "?" in merged
                    if question_mark_hit and not done_hit:
                        cleaned = merged.split("?", 1)[0].strip() + "?"
                    self._response_parts = []
                    self._capturing = False
                    self._last_transcript = ""
                    self._capture_timeout_paused = False
                    q = question_text_for_llm(cleaned, self.similarity_threshold)
                    if q:
                        response_callback_text = q
                    if not response_callback_text:
                        return
                # Append only the new part to response
                elif transcript.startswith(prev):
                    if not prev and wake_tail.strip():
                        new_bit = wake_tail
                    else:
                        new_bit = transcript[len(prev) :].lstrip()
                    if new_bit:
                        self._response_parts.append(new_bit)
                elif prev.startswith(transcript):
                    pass
                else:
                    # Full-replace caption (common with cloud ASR): strip any embedded wake
                    # so we do not accumulate "hey jetson … hey jetson what" twice.
                    tail = wake_tail
                    if tail.strip():
                        existing = " ".join(self._response_parts).strip()
                        if existing and tail.startswith(existing):
                            self._response_parts = [tail]
                        elif not (existing and existing.startswith(tail)):
                            self._response_parts.append(tail)
                    elif not _matches_trigger(transcript, self.similarity_threshold):
                        self._response_parts.append(transcript)
                merged = " ".join(self._response_parts).strip()
                cleaned, done_hit = self._pop_done_phrase(merged)
                question_mark_hit = "?" in merged
                if question_mark_hit and not done_hit:
                    cleaned = merged.split("?", 1)[0].strip() + "?"
                if done_hit or question_mark_hit:
                    self._response_parts = []
                    self._capturing = False
                    self._last_transcript = ""
                    self._capture_timeout_paused = False
                    q = question_text_for_llm(cleaned, self.similarity_threshold)
                    if q:
                        response_callback_text = q
                if not response_callback_text:
                    return

            if response_callback_text:
                pass
            elif self._can_wake is not None and not self._can_wake():
                self._last_transcript = ""
                return

            after = _text_after_trigger(transcript, self.similarity_threshold)
            has_post_wake_text = bool(after.strip())
            if not response_callback_text and (_matches_trigger(transcript, self.similarity_threshold) or has_post_wake_text):
                norm_full = _normalize(transcript)
                deb_key = _debounce_wake_key(norm_full)
                # Same wake line modulo punctuation (e.g. Deepgram alternates "hey, jetson." /
                # "hey, jetson"), nothing after the wake phrase, and still inside debounce window.
                skip_duplicate_wake = (
                    self._on_wake is not None
                    and not after.strip()
                    and deb_key
                    and deb_key == self._last_wake_debounce_key
                    and (now - self._last_on_wake_time) < self._wake_prompt_debounce_sec
                )
                if skip_duplicate_wake:
                    self._last_transcript = ""
                    return
                self._capturing = True
                self._response_parts = []
                if after:
                    self._response_parts.append(after)
                if after:
                    cleaned, done_hit = self._pop_done_phrase(after)
                    question_mark_hit = "?" in after
                    if question_mark_hit and not done_hit:
                        cleaned = after.split("?", 1)[0].strip() + "?"
                    if done_hit or question_mark_hit:
                        self._response_parts = []
                        self._capturing = False
                        self._last_transcript = ""
                        self._capture_timeout_paused = False
                        q = question_text_for_llm(cleaned, self.similarity_threshold)
                        if q:
                            response_callback_text = q
                if self._on_wake is not None and not skip_duplicate_wake and not after.strip():
                    wake_callback = self._on_wake
                    self._last_wake_debounce_key = deb_key
                    self._last_on_wake_time = now
                if self._capturing:
                    self._schedule_pause_check()
                # Baseline for the next caption chunk: only text after the wake, so
                # repeated "hey jetson" lines do not stack with the prior wake string.
                self._last_transcript = after
        if wake_callback is not None:
            try:
                wake_callback()
            except Exception:
                pass
        if response_callback_text:
            try:
                self.on_response(response_callback_text)
            except Exception:
                pass

    def stop(self) -> None:
        self._stop = True
        if self._timer:
            self._timer.cancel()
