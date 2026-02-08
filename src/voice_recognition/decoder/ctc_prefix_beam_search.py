"""CTC prefix beam search decoder (beam=8, no LM).

Implements Algorithm 2 from:
  https://www.merl.com/publications/docs/TR2017-190.pdf

Minimal dependencies: numpy only. Compatible with PyTorch/ONNX outputs
via .cpu().numpy() or equivalent.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np


def _logsumexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf and b == -math.inf:
        return -math.inf
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class CTCPrefixBeamSearch:
    """CTC prefix beam search decoder (no LM).

    Interface:
      decoder = CTCPrefixBeamSearch(vocab, blank_index=0, beam_size=8)
      text, score = decoder.decode(log_probs)

    Input: log_probs shape (T, V) - log probabilities per frame.
    Output: best hypothesis text and its log-score.
    """

    def __init__(
        self,
        vocab: List[str],
        blank_index: int = 0,
        beam_size: int = 8,
    ):
        """
        Args:
            vocab: List of symbols; index i maps to vocab[i].
            blank_index: Index of CTC blank token (default 0).
            beam_size: Beam width (default 8).
        """
        self.vocab = vocab
        self.blank_index = blank_index
        self.beam_size = beam_size
        self._minus_inf = -1e20

    def decode(
        self,
        log_probs: Union[np.ndarray, "np.ndarray"],
        seq_len: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Decode CTC log-probs to best hypothesis.

        Args:
            log_probs: Shape (T, V), log probabilities. Accepts numpy or torch.
            seq_len: Optional valid length T' <= T; if None, uses full T.

        Returns:
            (text, log_score): Best hypothesis string and its log-score.
        """
        lp = self._to_numpy(log_probs)
        T, V = lp.shape
        if seq_len is not None:
            T = min(T, int(seq_len))
        lp = lp[:T]

        # Beam: dict prefix -> (log_pb, log_pnb)
        # log_pb = prob prefix ends in blank, log_pnb = prob prefix ends in non-blank
        beam: dict[Tuple, Tuple[float, float]] = {
            (): (0.0, self._minus_inf)
        }

        for t in range(T):
            log_p_blank = lp[t, self.blank_index]
            next_beam: dict[Tuple, Tuple[float, float]] = {}

            for prefix, (log_pb, log_pnb) in beam.items():
                log_ptotal = _logsumexp(log_pb, log_pnb)

                # Blank: same prefix
                key = prefix
                pb_new, pnb_new = next_beam.get(key, (self._minus_inf, self._minus_inf))
                next_beam[key] = (
                    _logsumexp(pb_new, log_ptotal + log_p_blank),
                    pnb_new,
                )

                # Non-blank
                for c in range(V):
                    if c == self.blank_index:
                        continue
                    log_p_c = lp[t, c]
                    last = prefix[-1] if prefix else -1
                    if c == last:
                        # Extend run: same prefix, pnb += p(c) * pnb
                        key = prefix
                        pb_cur, pnb_cur = next_beam[key]
                        next_beam[key] = (
                            pb_cur,
                            _logsumexp(pnb_cur, log_pnb + log_p_c),
                        )
                    else:
                        # New token: prefix + c
                        new_prefix = prefix + (c,)
                        pb_cur, pnb_cur = next_beam.get(
                            new_prefix, (self._minus_inf, self._minus_inf)
                        )
                        next_beam[new_prefix] = (
                            pb_cur,
                            _logsumexp(pnb_cur, log_ptotal + log_p_c),
                        )

            # Prune to top beam_size by total score
            scored = [
                (prefix, _logsumexp(pb, pnb), pb, pnb)
                for prefix, (pb, pnb) in next_beam.items()
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            beam = {
                prefix: (pb, pnb)
                for prefix, _, pb, pnb in scored[: self.beam_size]
            }

        # Best hypothesis
        best_prefix = max(beam.keys(), key=lambda p: _logsumexp(beam[p][0], beam[p][1]))
        best_score = _logsumexp(beam[best_prefix][0], beam[best_prefix][1])
        text = self._indices_to_text(best_prefix)
        return text, best_score

    def _indices_to_text(self, indices: Tuple[int, ...]) -> str:
        """Convert index tuple to string."""
        return "".join(self.vocab[i] for i in indices)

    def _to_numpy(self, x: Union[np.ndarray, "object"]) -> np.ndarray:
        """Convert torch/numpy to numpy."""
        if isinstance(x, np.ndarray):
            return np.asarray(x, dtype=np.float32)
        # PyTorch
        if hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)


def ctc_greedy_decode(
    log_probs: np.ndarray,
    vocab: List[str],
    blank_index: int = 0,
    seq_len: Optional[int] = None,
) -> str:
    """Greedy CTC decode for comparison/validation.

    Args:
        log_probs: (T, V) log probabilities.
        vocab: Symbol list.
        blank_index: Blank token index.
        seq_len: Optional valid length.

    Returns:
        Decoded text string.
    """
    T = log_probs.shape[0] if seq_len is None else min(log_probs.shape[0], seq_len)
    pred = np.argmax(log_probs[:T], axis=1)
    out: List[int] = []
    prev = blank_index
    for i in range(T):
        if pred[i] != blank_index and pred[i] != prev:
            out.append(pred[i])
        prev = pred[i]
    return "".join(vocab[c] for c in out)
