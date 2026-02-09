"""Post-processing for decoder output (BPE, etc.)."""

import re


def refine_bpe_caption(raw: str) -> str:
    """Convert raw BPE/subword decoder output to readable caption text.

    Handles:
    - <?> placeholders (out-of-range indices)
    - ## WordPiece continuation marker
    - ▁ SentencePiece word-boundary marker
    - Multiple spaces
    """
    if not raw:
        return ""
    text = raw.replace("<?>", "")
    text = text.replace("##", "")
    text = text.replace("▁", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
