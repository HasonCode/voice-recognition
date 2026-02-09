"""
Listen for "hey jetson" (with fuzzy + regex match), then capture the following
speech until a pause and feed it to a callback.

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

# Regex: "hey" then optional space then "jetson" at end of text
TRIGGER_REGEX = re.compile(
    r"\bhey\s*jetson\s*$",
    re.IGNORECASE,
)

# Minimum similarity (0–1) for fuzzy match on last words
DEFAULT_SIMILARITY_THRESHOLD = 0.75

# Seconds of no new transcript to consider "pause in speech"
DEFAULT_PAUSE_SEC = 2.5

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


def _matches_trigger(text: str, threshold: float) -> bool:
    """True if end of text matches 'hey jetson' via regex or fuzzy match."""
    if not text or not text.strip():
        return False
    norm = _normalize(text)
    # Regex: end of string matches "hey" + optional space + jetson-like
    if TRIGGER_REGEX.search(norm):
        return True
    # Fuzzy: compare last 2–4 words to trigger phrase
    trigger_words = TRIGGER_PHRASE.split()
    for n in (2, 3, 4):
        suffix = _last_n_words(norm, n)
        if _similarity(suffix, TRIGGER_PHRASE) >= threshold:
            return True
        # Allow "hey jetson" as substring of last n words
        if TRIGGER_PHRASE in suffix or _similarity(suffix, TRIGGER_PHRASE) >= threshold:
            return True
    return False


def _text_after_trigger(text: str) -> str:
    """Return the part of text after the last occurrence of trigger-like phrase."""
    norm = _normalize(text)
    m = re.search(r"\bhey\s*jetson\b", norm, re.IGNORECASE)
    if m:
        return norm[m.end() :].strip()
    return ""


class HeyJetsonListener:
    """
    Listens to a stream of transcript updates. When "hey jetson" is detected at
    the end of the transcript, captures all following speech until a pause
    (no new content for pause_sec), then calls on_response with that text.
    """

    def __init__(
        self,
        on_response: Callable[[str], None],
        *,
        trigger_phrase: str = TRIGGER_PHRASE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        pause_sec: float = DEFAULT_PAUSE_SEC,
    ):
        self.on_response = on_response
        self.similarity_threshold = similarity_threshold
        self.pause_sec = pause_sec
        self._trigger_phrase = _normalize(trigger_phrase)

        self._last_transcript = ""
        self._last_update_time: float = 0.0
        self._capturing = False
        self._response_parts: list[str] = []
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._stop = False

    def _schedule_pause_check(self) -> None:
        def check():
            if self._stop:
                return
            with self._lock:
                if not self._capturing:
                    return
                elapsed = time.time() - self._last_update_time
                if elapsed >= self.pause_sec and self._response_parts:
                    response = " ".join(self._response_parts).strip()
                    self._response_parts = []
                    self._capturing = False
                    try:
                        self.on_response(response)
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
                return

            if _matches_trigger(transcript, self.similarity_threshold):
                self._capturing = True
                self._response_parts = []
                after = _text_after_trigger(transcript)
                if after:
                    self._response_parts.append(after)
                self._schedule_pause_check()

    def stop(self) -> None:
        self._stop = True
        if self._timer:
            self._timer.cancel()
