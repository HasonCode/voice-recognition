"""Unit tests and toy examples for CTC prefix beam search decoder."""

from __future__ import annotations

import unittest

import numpy as np

from voice_recognition.decoder.ctc_prefix_beam_search import (
    CTCPrefixBeamSearch,
    ctc_greedy_decode,
)


# Toy vocab: blank, a, b, c, space
VOCAB = ["<blank>", "a", "b", "c", " "]
BLANK = 0


class TestCTCPrefixBeamSearch(unittest.TestCase):
    """Tests for CTC prefix beam search."""

    def setUp(self) -> None:
        self.decoder = CTCPrefixBeamSearch(VOCAB, blank_index=BLANK, beam_size=8)

    def test_empty_log_probs(self) -> None:
        """Empty input should return empty string."""
        log_probs = np.zeros((0, len(VOCAB)), dtype=np.float32)
        text, score = self.decoder.decode(log_probs)
        self.assertEqual(text, "")
        self.assertTrue(score <= 0)

    def test_all_blank(self) -> None:
        """All-blank frames should produce empty string."""
        T = 10
        log_probs = np.full((T, len(VOCAB)), -10.0, dtype=np.float32)
        log_probs[:, BLANK] = 0.0
        text, _ = self.decoder.decode(log_probs)
        self.assertEqual(text, "")

    def test_single_token(self) -> None:
        """Single non-blank frame should produce that token."""
        log_probs = np.full((5, len(VOCAB)), -10.0, dtype=np.float32)
        log_probs[:, BLANK] = -1.0
        log_probs[2, 1] = 0.0  # 'a' at frame 2
        text, _ = self.decoder.decode(log_probs)
        self.assertEqual(text, "a")

    def test_sequence_abc(self) -> None:
        """Clear abc sequence should decode to 'abc'."""
        T = 12
        log_probs = np.full((T, len(VOCAB)), -10.0, dtype=np.float32)
        log_probs[:, BLANK] = -2.0
        log_probs[1, 1] = 0.0   # a
        log_probs[4, 2] = 0.0   # b
        log_probs[7, 3] = 0.0   # c
        text, _ = self.decoder.decode(log_probs)
        self.assertEqual(text, "abc")

    def test_repeated_characters(self) -> None:
        """Repeated chars (aa) should collapse to 'a'."""
        log_probs = np.full((6, len(VOCAB)), -10.0, dtype=np.float32)
        log_probs[:, BLANK] = -2.0
        log_probs[1, 1] = 0.0
        log_probs[2, 1] = 0.0
        log_probs[3, 1] = 0.0
        text, _ = self.decoder.decode(log_probs)
        self.assertEqual(text, "a")

    def test_beam_size_effect(self) -> None:
        """Beam size 1 should match greedy; beam 8 should be >= greedy score."""
        np.random.seed(42)
        T, V = 20, len(VOCAB)
        log_probs = np.random.randn(T, V).astype(np.float32) * 0.5
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=1, keepdims=True))

        greedy_text = ctc_greedy_decode(log_probs, VOCAB, BLANK)
        _, beam1_score = CTCPrefixBeamSearch(VOCAB, BLANK, beam_size=1).decode(log_probs)
        _, beam8_score = CTCPrefixBeamSearch(VOCAB, BLANK, beam_size=8).decode(log_probs)

        # Beam 8 should have score >= beam 1 (beam 1 is equivalent to greedy)
        self.assertGreaterEqual(beam8_score, beam1_score - 1e-5)

    def test_interface_numpy(self) -> None:
        """Should accept numpy array."""
        log_probs = np.zeros((3, len(VOCAB)), dtype=np.float32)
        log_probs[1, 1] = 0.0
        text, _ = self.decoder.decode(log_probs)
        self.assertEqual(text, "a")

    def test_interface_seq_len(self) -> None:
        """seq_len should limit decoding."""
        log_probs = np.full((10, len(VOCAB)), -10.0, dtype=np.float32)
        log_probs[:, BLANK] = -1.0
        log_probs[1, 1] = 0.0
        log_probs[8, 2] = 0.0
        text_full, _ = self.decoder.decode(log_probs)
        text_partial, _ = self.decoder.decode(log_probs, seq_len=5)
        self.assertEqual(text_full, "ab")
        self.assertEqual(text_partial, "a")


def run_toy_example() -> None:
    """Toy example: decode synthetic CTC output."""
    print("=== Toy example: CTC prefix beam search ===\n")
    vocab = ["<blank>", "h", "e", "l", "o"]
    decoder = CTCPrefixBeamSearch(vocab, blank_index=0, beam_size=8)

    # Synthetic: "hello" with blanks and repeats
    T = 20
    log_probs = np.full((T, len(vocab)), -5.0, dtype=np.float32)
    log_probs[:, 0] = -0.5  # blank somewhat likely
    # h e l l o
    log_probs[2, 1] = 0.0
    log_probs[5, 2] = 0.0
    log_probs[7, 3] = 0.0
    log_probs[8, 3] = 0.0
    log_probs[12, 4] = 0.0

    text, score = decoder.decode(log_probs)
    greedy = ctc_greedy_decode(log_probs, vocab, 0)
    print(f"Beam search: '{text}' (log-score: {score:.2f})")
    print(f"Greedy:      '{greedy}'")
    print("\nDone.")


if __name__ == "__main__":
    run_toy_example()
    print("\n--- Running unit tests ---")
    unittest.main(argv=[""], exit=False, verbosity=2)
