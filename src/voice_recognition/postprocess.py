"""Post-processing for decoder output (BPE, etc.)."""

import re

# Common word starts for heuristic spacing (insert space before when preceded by letter)
_WORD_STARTS = (
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "to", "from", "the", "a", "an", "and", "or", "of", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "can", "may",
    "for", "with", "by", "as", "it", "its", "this", "that", "these", "those",
)


def refine_bpe_caption(raw: str) -> str:
    """Convert raw BPE/subword decoder output to readable caption text.

    Handles:
    - <?> placeholders (out-of-range indices)
    - ## WordPiece continuation marker
    - ▁ SentencePiece word-boundary marker
    - Heuristic word spacing (insert space before common words)
    - Multiple spaces
    - Consecutive duplicate words (CTC / overlapping windows)
    """
    if not raw:
        return ""
    text = raw.replace("<?>", "")
    text = text.replace("##", "")
    text = text.replace("▁", " ")
    text = _add_word_spaces(text)
    text = re.sub(r"\s+", " ", text)
    text = _deduplicate_consecutive_words(text)
    return text.strip()


def _deduplicate_consecutive_words(text: str) -> str:
    """Remove consecutive duplicate words (e.g. 'hello hello world' -> 'hello world')."""
    words = text.split()
    if not words:
        return ""
    result = [words[0]]
    for w in words[1:]:
        if w != result[-1]:
            result.append(w)
    return " ".join(result)


def merge_display_into_transcript(accumulated: str, new_display: str) -> str:
    """Merge new display into accumulated transcript using word-boundary overlap.

    For overlapping windows, finds the longest word overlap between the end of
    accumulated and the start of new_display, then appends the new part.
    E.g. accumulated='one two three', new_display='two three four five'
    -> overlap 'two three', append 'four five' -> 'one two three four five'.

    When no overlap, keeps the longer of the two (captures more content).
    """
    if not new_display:
        return accumulated
    if not accumulated:
        return new_display.strip()

    acc_words = accumulated.split()
    new_words = new_display.split()

    # Find longest overlap: words at end of accumulated matching words at start of new
    best_overlap = 0
    for overlap in range(1, min(len(acc_words), len(new_words)) + 1):
        if acc_words[-overlap:] == new_words[:overlap]:
            best_overlap = overlap

    if best_overlap > 0:
        suffix = new_words[best_overlap:]
        if suffix:
            return accumulated + " " + " ".join(suffix)
        return accumulated

    # No overlap: keep the longer segment (avoids losing content from window jumps)
    return accumulated if len(accumulated) >= len(new_display) else new_display.strip()


def _add_word_spaces(text: str) -> str:
    """Insert space before common word starts when preceded by a letter."""
    result = text
    for w in sorted(_WORD_STARTS, key=len, reverse=True):
        # Space before word when preceded by letter (e.g. fromone -> from one)
        pattern = r"([a-zA-Z])(" + re.escape(w) + r")(?=[a-zA-Z]|$)"
        result = re.sub(pattern, r"\1 \2", result, flags=re.IGNORECASE)
    return result
