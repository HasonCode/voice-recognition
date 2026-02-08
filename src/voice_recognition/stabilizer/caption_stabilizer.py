"""Caption stabilizer: common-prefix + N-stable-updates commit logic.

Reduces flicker in streaming captions by only committing text that has
remained unchanged for N consecutive decoder updates.

Pipeline: decoder -> stabilizer -> display

Minimal deps: none (stdlib only).
"""

from __future__ import annotations


class CaptionStabilizer:
    """Stabilize streaming partial transcripts using N-stable commit logic.

    - Committed: text already confirmed and shown.
    - Tentative: suffix from latest partial that extends committed; shown
      but not yet committed.
    - When tentative is unchanged for `stable_n` consecutive updates, it
      is committed (moved into committed).

    Interface:
      stabilizer = CaptionStabilizer(stable_n=2)
      display_text = stabilizer.update(partial_from_decoder)
      stabilizer.commit()   # optional: force-commit current tentative (e.g. end of utterance)
      stabilizer.reset()   # optional: new segment
    """

    def __init__(self, stable_n: int = 2):
        """
        Args:
            stable_n: Number of consecutive updates the same tentative
                      must appear before it is committed (default 2).
        """
        if stable_n < 1:
            raise ValueError("stable_n must be >= 1")
        self.stable_n = stable_n
        self._committed = ""
        self._last_tentative = ""
        self._stable_count = 0

    def update(self, partial: str) -> str:
        """Update with new partial from decoder; return string to display.

        Args:
            partial: Current best hypothesis from CTC decoder.

        Returns:
            Text to display: committed + tentative (tentative only if
            partial extends committed; otherwise committed only).
        """
        if not partial:
            return self._committed

        if not self._committed or partial.startswith(self._committed):
            # Partial extends committed (or we have no commitment yet)
            tentative = partial[len(self._committed) :]
            if tentative == self._last_tentative:
                self._stable_count += 1
                if self._stable_count >= self.stable_n and tentative:
                    self._committed = partial
                    self._last_tentative = ""
                    self._stable_count = 0
                    return self._committed
            else:
                self._last_tentative = tentative
                self._stable_count = 1
            return self._committed + tentative

        # Partial does not extend committed (decoder changed mind)
        # Keep committed; do not show conflicting text until decoder agrees again
        self._last_tentative = ""
        self._stable_count = 0
        return self._committed

    def commit(self) -> str:
        """Force-commit current tentative (e.g. on end-of-utterance).

        Returns:
            New committed string (committed + previous tentative).
        """
        if self._last_tentative:
            self._committed = self._committed + self._last_tentative
            self._last_tentative = ""
            self._stable_count = 0
        return self._committed

    def reset(self) -> None:
        """Reset for a new segment (e.g. new utterance)."""
        self._committed = ""
        self._last_tentative = ""
        self._stable_count = 0

    @property
    def committed(self) -> str:
        """Currently committed text."""
        return self._committed

    @property
    def tentative(self) -> str:
        """Current tentative suffix (not yet committed)."""
        return self._last_tentative
