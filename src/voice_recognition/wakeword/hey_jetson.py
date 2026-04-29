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
    r"\bhey\s*jetson\s*$",
    re.IGNORECASE,
)

# ASR often splits/glues differently than "hey jetson" — match common variants at **end** of transcript.
_WAKE_AT_END = re.compile(
    r"(?:"
    r"\bhey\s*jetson"
    r"|\bhey\s+jets?\s+on"
    r"|\bheyjets?\s+on"
    r"|\bh\s+ay\s+jets?\s+on"
    r"|\bhe+jets?\s+on"
    r"|\bhey\s+jet\s*son"
    r"|\bhey\s+jetds\s+on"
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
DEFAULT_DONE_PHRASES = (
    "jets on done",
    "jetson done",
    "jet done",
    "that is all",
    "thank you",
    "done",
    "over",
)
# Tolerant end-of-question variants for noisy ASR, e.g. "jet son done", "jetsen done",
# "jet done", "jets on over", or plain "done"/"over".
_DONE_TAIL_REGEX = re.compile(
    r"(?:"
    r"\b(?:jet\s*son|jets?\s*on|jetsen|jetson)\s+"
    r"(?:done|over|that\s+is\s+all|thank\s+you)"
    r"|"
    r"\b(?:jet|jets)\s+(?:done|over|that\s+is\s+all|thank\s+you)"
    r"|"
    r"\b(?:done|over|that\s+is\s+all|thank\s+you)"
    r")\s*$",
    re.IGNORECASE,
)

# How often to check for pause (sec)
PAUSE_CHECK_INTERVAL = 0.3


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip()


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


def _matches_trigger(text: str, threshold: float) -> bool:
    """True if end of text matches 'hey jetson' or short 'hey jet' via regex or fuzzy match."""
    if not text or not text.strip():
        return False
    norm = _normalize(text)
    # Regex: canonical or ASR-variant wake at end of string
    if TRIGGER_REGEX.search(norm) or _WAKE_AT_END.search(norm) or _HEY_JET_AT_END.search(norm):
        return True
    # Fuzzy: compare last 2–5 words to trigger phrase (word-level)
    for n in (2, 3, 4, 5):
        suffix = _last_n_words(norm, n)
        if not suffix:
            continue
        if suffix == "hey jet" or _similarity(suffix, "hey jet") >= min(0.88, threshold + 0.1):
            return True
        if suffix.endswith(TRIGGER_PHRASE) or _similarity(suffix, TRIGGER_PHRASE) >= threshold:
            return True
    # Letters-only tail: "heyjets on", "h ayjets on", "hejets on" → heyjetson
    tail_words = _last_n_words(norm, 6)
    lt = _letters_only(tail_words)
    if _letter_tail_wake_score(lt) >= _letter_wake_threshold(threshold):
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
        re.compile(r"\bhey\s*jetson\b", re.IGNORECASE),
        re.compile(
            r"\bhey\s+jets?\s+on\b|\bheyjets?\s+on\b|\bh\s+ay\s+jets?\s+on\b|\bhe+jets?\s+on\b"
            r"|\bhey\s+jet\s*son\b|\bhey\s+jetds\s+on\b",
            re.IGNORECASE,
        ),
        # Short wake; do not match the "hey jet" prefix of "hey jet son" (common ASR for jetson).
        re.compile(r"\bhey\s+jet\b(?!\s*son\b)", re.IGNORECASE),
    ):
        last_end = -1
        for m in pat.finditer(norm):
            last_end = m.end()
        if last_end >= 0:
            return norm[last_end:].strip()
    stripped = _strip_trailing_wake_words(norm, similarity_threshold)
    if stripped is not None:
        return stripped
    return ""


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
    ):
        self.on_response = on_response
        self.similarity_threshold = similarity_threshold
        self.pause_sec = pause_sec
        self.done_phrases = tuple(_normalize(x) for x in done_phrases if _normalize(x))
        self._on_wake = on_wake
        self._trigger_phrase = _normalize(trigger_phrase)

        self._last_transcript = ""
        self._last_update_time: float = 0.0
        self._capturing = False
        self._response_parts: list[str] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._stop = False

    def _pop_done_phrase(self, text: str) -> tuple[str, bool]:
        """Strip a trailing done phrase from text. Returns (clean_text, matched)."""
        norm = _normalize(text)
        if not norm:
            return "", False
        # Try tolerant/noisy-ASR endings first so "jet son done" strips as a unit
        # before generic one-word endings like "done"/"over".
        m = _DONE_TAIL_REGEX.search(norm)
        if m is not None:
            trimmed = norm[: m.start()].strip()
            return trimmed, True
        for phrase in sorted(self.done_phrases, key=len, reverse=True):
            if norm.endswith(phrase):
                trimmed = norm[: -len(phrase)].strip()
                return trimmed, True
        return norm, False

    def _schedule_pause_check(self) -> None:
        def check():
            if self._stop:
                return
            with self._lock:
                if not self._capturing:
                    return
                elapsed = time.time() - self._last_update_time
                if elapsed >= self.pause_sec:
                    # Timeout capture even when no words followed wake phrase; otherwise
                    # we can get stuck in capture mode forever after a false wake.
                    if self._response_parts:
                        response = " ".join(self._response_parts).strip()
                        response, _ = self._pop_done_phrase(response)
                        self._response_parts = []
                        self._capturing = False
                        if response:
                            try:
                                self.on_response(response)
                            except Exception:
                                pass
                        return
                    self._capturing = False
                    self._response_parts = []
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
        with self._lock:
            now = time.time()
            prev = self._last_transcript
            self._last_transcript = transcript
            self._last_update_time = now

            if self._capturing:
                # Append only the new part to response
                if transcript.startswith(prev):
                    new_bit = transcript[len(prev) :].lstrip()
                    if new_bit:
                        self._response_parts.append(new_bit)
                elif prev.startswith(transcript):
                    pass
                else:
                    self._response_parts.append(transcript)
                merged = " ".join(self._response_parts).strip()
                cleaned, done_hit = self._pop_done_phrase(merged)
                if done_hit:
                    self._response_parts = []
                    self._capturing = False
                    if cleaned:
                        try:
                            self.on_response(cleaned)
                        except Exception:
                            pass
                return

            if _matches_trigger(transcript, self.similarity_threshold):
                self._capturing = True
                self._response_parts = []
                after = _text_after_trigger(transcript, self.similarity_threshold)
                if after:
                    self._response_parts.append(after)
                wake_callback = self._on_wake
                self._schedule_pause_check()
        if wake_callback is not None:
            try:
                wake_callback()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop = True
        if self._timer:
            self._timer.cancel()
