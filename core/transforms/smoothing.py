"""Smoothing — replace the series with a rolling mean."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform, TransformError
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class Smoothing(Transform):
    """Apply a trailing rolling mean.

    Args:
        window: Number of points averaged into each plotted point.
    """

    window: int = 7

    name = "smoothing"
    label = "Smoothed series"
    explanation = "Plots a trailing rolling mean instead of the raw values."
    why_misleading = (
        "Smoothing removes the volatility a reader would use to judge whether "
        "a trend is real. A wide enough window turns noise into a clean line "
        "pointing wherever the last few points happen to sit."
    )
    legitimate_when = (
        "Seasonality or reporting artifacts genuinely obscure the trend — a "
        "7-day mean on weekday-seasonal data — and the window is disclosed."
    )

    def __post_init__(self) -> None:
        if self.window < 2:
            raise TransformError("a smoothing window must cover at least two points")

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        return frame, self._stamp(spec, smoothing_window=self.window)
