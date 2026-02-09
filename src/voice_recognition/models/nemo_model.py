"""NeMo ASR model loader for streaming caption pipeline.

NeMo models expect raw audio (not mel); the model performs its own
preprocessing. Use with pipeline model_input="audio".
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _get_nemo():
    import nemo.collections.asr as nemo_asr
    return nemo_asr


def load_nemo_model(
    path: str | Path,
    device: Optional[str] = None,
) -> Tuple["callable", List[str], int]:
    """Load a NeMo CTC ASR model and return a forward callable plus vocab.

    Args:
        path: Path to .nemo checkpoint file.
        device: Optional device string ('cuda', 'cpu', etc.). If None,
                uses CUDA if available else CPU.

    Returns:
        (model_forward, vocab, blank_index):
        - model_forward(audio: np.ndarray) -> log_probs: np.ndarray
          audio shape (samples,) float32 mono 16 kHz
          log_probs shape (T, V)
        - vocab: list of symbols
        - blank_index: CTC blank token index (usually 0)
    """
    import torch

    nemo_asr = _get_nemo()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NeMo model not found: {path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(path),
        map_location=torch.device(device),
    )
    model.eval()

    vocab = _extract_vocab(model)
    blank_index = 0

    def forward(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            audio_t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            length_t = torch.tensor([audio_t.shape[1]], dtype=torch.int64, device=audio_t.device)
            audio_t = audio_t.to(model.device)
            length_t = length_t.to(model.device)
            log_probs, _, _ = model.forward(
                input_signal=audio_t,
                input_signal_length=length_t,
            )
            return log_probs.squeeze(0).cpu().numpy().astype(np.float32)

    return forward, vocab, blank_index


def _extract_vocab(model) -> List[str]:
    """Extract vocabulary list from NeMo ASR model."""
    if hasattr(model, "decoder") and hasattr(model.decoder, "vocabulary"):
        v = model.decoder.vocabulary
        if isinstance(v, list):
            return list(v)
        if hasattr(v, "copy"):
            return list(v.copy())
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        # BPE/subword tokenizer
        if hasattr(model.tokenizer, "vocab"):
            return list(model.tokenizer.vocab.keys())
        if hasattr(model.tokenizer, "ids_to_tokens"):
            n = getattr(model.tokenizer, "vocab_size", 1024)
            return [model.tokenizer.ids_to_tokens(i) for i in range(n)]
    raise ValueError("Could not extract vocabulary from NeMo model")
