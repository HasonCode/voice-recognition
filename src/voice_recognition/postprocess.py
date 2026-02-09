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
    """
    if not raw:
        return ""
    text = raw.replace("<?>", "")
    text = text.replace("##", "")
    text = text.replace("▁", " ")
    text = _add_word_spaces(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _add_word_spaces(text: str) -> str:
    """Insert space before common word starts when preceded by a letter."""
    result = text
    for w in sorted(_WORD_STARTS, key=len, reverse=True):
        # Space before word when preceded by letter (e.g. fromone -> from one)
        pattern = r"([a-zA-Z])(" + re.escape(w) + r")(?=[a-zA-Z]|$)"
        result = re.sub(pattern, r"\1 \2", result, flags=re.IGNORECASE)
    return result
