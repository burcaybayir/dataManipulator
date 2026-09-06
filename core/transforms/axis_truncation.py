"""Truncated axis — start the y axis above zero."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform, TransformError, prepared_or_error
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class TruncatedAxis(Transform):
    """Raise the y-axis floor to just below the smallest plotted value.

    Args:
        floor: Explicit y-axis minimum. When ``None`` it is computed from the
            data as ``min - headroom * (max - min)``.
        headroom: Slack below the smallest value, as a share of the plotted
            range. Smaller values make the distortion more extreme.
    """

    floor: float | None = None
    headroom: float = 0.05

    name = "truncated_axis"
    label = "Truncated axis"
    explanation = "Starts the y axis just below the lowest plotted value instead of at zero."
    why_misleading = (
        "The reader judges the size of a change by how far the mark moves. "
        "Truncating the axis magnifies small differences into dramatic ones."
    )
    legitimate_when = (
        "The quantity has a meaningful non-zero baseline — body temperature, "
        "a rate that never approaches zero — and the axis is labeled clearly."
    )

    def __post_init__(self) -> None:
        if self.headroom < 0:
            raise TransformError("headroom cannot be negative")

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        floor = self.floor if self.floor is not None else self._computed_floor(frame, spec)
        if spec.y_max is not None and floor >= spec.y_max:
            raise TransformError(f"floor {floor} is at or above the axis maximum {spec.y_max}")
        if floor <= prepared_or_error(frame, spec).y_domain[0]:
            raise TransformError(
                f"a floor of {floor:.4g} would not truncate anything: the series already "
                "reaches the bottom of its axis"
            )
        return frame, self._stamp(spec, y_min=floor)

    def _computed_floor(self, frame: pd.DataFrame, spec: ChartSpec) -> float:
        values = prepared_or_error(frame, spec).values.dropna()
        if values.empty:
            raise TransformError("no values to truncate against")
        low, high = float(values.min()), float(values.max())
        span = high - low
        return low - (span * self.headroom if span else abs(low) * self.headroom)
