"""Cherry-picked window — show only the stretch that supports the claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform, TransformError, prepared_or_error
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class CherryPickedWindow(Transform):
    """Restrict the chart to an x range.

    Args:
        start: Inclusive lower bound, or ``None`` for the earliest point.
        end: Inclusive upper bound, or ``None`` for the latest point.

    Use :meth:`favouring` to let the transform find the window that best
    supports a direction rather than naming one.
    """

    start: Any = None
    end: Any = None

    name = "cherry_picked_window"
    label = "Cherry-picked window"
    explanation = "Restricts the chart to a chosen slice of the x axis."
    why_misleading = (
        "Almost any series contains a stretch that rises and a stretch that "
        "falls. Choosing one and omitting the rest lets the same data support "
        "either conclusion."
    )
    legitimate_when = (
        "The window is the question — a launch, a policy change, the current "
        "fiscal year — and the boundaries are stated up front."
    )

    def __post_init__(self) -> None:
        if self.start is None and self.end is None:
            raise TransformError("a window needs a start, an end, or both")

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        return frame, self._stamp(spec, x_range=(self.start, self.end))

    @classmethod
    def favouring(
        cls,
        frame: pd.DataFrame,
        spec: ChartSpec,
        *,
        direction: str,
        length: int = 6,
    ) -> CherryPickedWindow:
        """Find the ``length``-point window with the strongest trend.

        Args:
            direction: ``"up"`` or ``"down"`` — the story to support.
            length: Number of x points the window should span.

        Raises:
            TransformError: The direction is not recognized, the length is
                below two points, or the data is too short to window.
        """
        if direction not in {"up", "down"}:
            raise TransformError("direction must be 'up' or 'down'")
        if length < 2:
            raise TransformError("a window needs at least two points")

        totals = prepared_or_error(frame, spec).frame.groupby("x", observed=True)["value"].sum()
        totals = totals.sort_index()
        if len(totals) < length:
            raise TransformError(f"only {len(totals)} points available, need {length}")

        sign = 1.0 if direction == "up" else -1.0
        best_start, best_score = 0, float("-inf")
        for i in range(len(totals) - length + 1):
            window = totals.iloc[i : i + length]
            score = sign * float(window.iloc[-1] - window.iloc[0])
            if score > best_score:
                best_start, best_score = i, score

        chosen = totals.index[best_start : best_start + length]
        return cls(start=_label(chosen[0]), end=_label(chosen[-1]))


def _label(value: Any) -> Any:
    """Render an x value as something a manifest can hold."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
