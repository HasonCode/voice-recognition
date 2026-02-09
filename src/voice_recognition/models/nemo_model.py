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

    # Ensure vocab length matches model output: run minimal forward to get V
    def _get_output_vocab_size():
        with torch.no_grad():
            dummy = torch.zeros(1, 1600, device=model.device)
            log_probs, _, _ = model.forward(
                input_signal=dummy,
                input_signal_length=torch.tensor([1600], dtype=torch.int64, device=model.device),
            )
            return log_probs.shape[-1]

    try:
        V = _get_output_vocab_size()
        while len(vocab) < V:
            vocab.append("")
    except Exception:
        pass

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
    """Extract vocabulary list from NeMo ASR model.

    Returns vocab where vocab[i] = symbol for model output index i.
    Length must match model output dimension (log_probs.shape[1]).
    """
    vocab_size = None
    vocab: List[str] = []

    # 1. Try decoder.vocabulary (character CTC)
    if hasattr(model, "decoder") and hasattr(model.decoder, "vocabulary"):
        v = model.decoder.vocabulary
        if isinstance(v, (list, tuple)):
            vocab = list(v)
            vocab_size = len(vocab)
        elif hasattr(v, "__len__") and hasattr(v, "__getitem__"):
            vocab = [v[i] for i in range(len(v))]
            vocab_size = len(vocab)

    # 2. Try tokenizer (BPE/subword) - build by id order
    if (not vocab or vocab_size is None) and hasattr(model, "tokenizer") and model.tokenizer is not None:
        tok = model.tokenizer
        n = getattr(tok, "vocab_size", None) or getattr(tok, "get_vocab_size", lambda: 0)()
        if n == 0 and hasattr(tok, "get_vocab"):
            n = len(tok.get_vocab())
        if n > 0 and hasattr(tok, "ids_to_tokens"):
            vocab = [tok.ids_to_tokens(i) for i in range(n)]
            vocab_size = n

    # 3. Fallback: infer vocab_size from decoder output dim and pad if needed
    if not vocab and hasattr(model, "decoder"):
        # Decoder linear out features = num classes
        for attr in ("vocab_size", "num_classes", "_vocab_size"):
            if hasattr(model.decoder, attr):
                n = getattr(model.decoder, attr)
                if callable(n):
                    n = n()
                if isinstance(n, int) and n > 0:
                    vocab_size = n
                    break
        if vocab_size and hasattr(model, "tokenizer") and model.tokenizer is not None:
            tok = model.tokenizer
            if hasattr(tok, "ids_to_tokens"):
                vocab = [tok.ids_to_tokens(i) for i in range(vocab_size)]

    if not vocab:
        raise ValueError("Could not extract vocabulary from NeMo model")
    return vocab
