"""Unit tests and toy example for caption stabilizer (commit logic)."""

from __future__ import annotations

import unittest

from voice_recognition.stabilizer import CaptionStabilizer


class TestCaptionStabilizer(unittest.TestCase):
    """Tests for CaptionStabilizer."""

    def test_stable_n_invalid(self) -> None:
        """stable_n must be >= 1."""
        with self.assertRaises(ValueError):
            CaptionStabilizer(stable_n=0)

    def test_empty_partials(self) -> None:
        """Empty partials return committed only (no new tentative)."""
        s = CaptionStabilizer(stable_n=2)
        self.assertEqual(s.update(""), "")
        self.assertEqual(s.update("hello"), "hello")
        self.assertEqual(s.update("hello"), "hello")
        self.assertEqual(s.update(""), "hello")

    def test_commit_after_n_stable(self) -> None:
        """Tentative is committed after stable_n identical updates."""
        s = CaptionStabilizer(stable_n=2)
        self.assertEqual(s.update("ab"), "ab")
        self.assertEqual(s.update("abc"), "abc")
        self.assertEqual(s.update("abcd"), "abcd")
        self.assertEqual(s.update("abcd"), "abcd")
        self.assertEqual(s.committed, "abcd")
        self.assertEqual(s.tentative, "")

    def test_no_commit_until_stable(self) -> None:
        """With stable_n=2, one update is not enough to commit."""
        s = CaptionStabilizer(stable_n=2)
        self.assertEqual(s.update("a"), "a")
        self.assertEqual(s.committed, "")
        self.assertEqual(s.tentative, "a")
        self.assertEqual(s.update("a"), "a")
        self.assertEqual(s.committed, "a")
        self.assertEqual(s.tentative, "")

    def test_flicker_suppression(self) -> None:
        """Changing partial resets stable count; no commit until stable again."""
        s = CaptionStabilizer(stable_n=2)
        s.update("hello")
        s.update("hello")
        self.assertEqual(s.committed, "hello")
        s.update("hell")
        self.assertEqual(s.update("hell"), "hello")
        self.assertEqual(s.committed, "hello")
        s.update("hello")
        s.update("hello")
        self.assertEqual(s.committed, "hello")

    def test_decoder_backtrack(self) -> None:
        """When partial no longer extends committed, display stays committed only."""
        s = CaptionStabilizer(stable_n=2)
        s.update("abc")
        s.update("abc")
        self.assertEqual(s.committed, "abc")
        self.assertEqual(s.update("ab"), "abc")
        self.assertEqual(s.update("ab"), "abc")
        self.assertEqual(s.committed, "abc")

    def test_force_commit(self) -> None:
        """commit() forces current tentative into committed."""
        s = CaptionStabilizer(stable_n=3)
        s.update("hi")
        s.update("hi there")
        self.assertEqual(s.committed, "")
        self.assertEqual(s.tentative, "hi there")
        s.commit()
        self.assertEqual(s.committed, "hi there")
        self.assertEqual(s.tentative, "")

    def test_reset(self) -> None:
        """reset() clears committed and tentative."""
        s = CaptionStabilizer(stable_n=2)
        s.update("old")
        s.update("old")
        s.reset()
        self.assertEqual(s.committed, "")
        self.assertEqual(s.tentative, "")
        self.assertEqual(s.update("new"), "new")

    def test_stable_n_one(self) -> None:
        """stable_n=1 commits on first repeat (same as immediate commit on second occurrence)."""
        s = CaptionStabilizer(stable_n=1)
        self.assertEqual(s.update("x"), "x")
        self.assertEqual(s.update("x"), "x")
        self.assertEqual(s.committed, "x")


def run_toy_example() -> None:
    """Simulate streaming decoder updates and show stabilizer output."""
    print("=== Toy example: caption stabilizer (stable_n=2) ===\n")
    s = CaptionStabilizer(stable_n=2)
    stream = ["h", "he", "hel", "hell", "hello", "hello", "hello world", "hello world"]
    for i, partial in enumerate(stream):
        display = s.update(partial)
        print(f"  partial: {partial!r:20} -> display: {display!r}  committed: {s.committed!r}")
    print("\nDone.")


if __name__ == "__main__":
    run_toy_example()
    print("\n--- Running unit tests ---")
    unittest.main(argv=[""], exit=False, verbosity=2)
