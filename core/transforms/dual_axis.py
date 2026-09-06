"""Dual axis — put a second measure on its own scale to imply correlation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform, TransformError
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class DualAxis(Transform):
    """Render extra measure columns against an independent y axis.

    Args:
        columns: Measure columns to move onto the secondary axis. When empty,
            the first numeric column that is not already plotted is used.
    """

    columns: tuple[str, ...] = ()

    name = "dual_axis"
    label = "Dual axis"
    explanation = "Draws a second measure on its own y scale in the same plot frame."
    why_misleading = (
        "Two independent scales can be slid until unrelated series appear to "
        "move together. The apparent correlation is a choice of scale, not a "
        "property of the data."
    )
    legitimate_when = (
        "Two series in different units are genuinely being compared over time "
        "and both axes are labeled and colour-matched to their series."
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        columns = self.columns or (self._first_spare_measure(frame, spec),)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise TransformError(f"columns not in the data: {', '.join(missing)}")
        return frame, self._stamp(spec, secondary_y=tuple(columns))

    def _first_spare_measure(self, frame: pd.DataFrame, spec: ChartSpec) -> str:
        plotted = set(spec.columns)
        for column in frame.columns:
            if column not in plotted and pd.api.types.is_numeric_dtype(frame[column]):
                return str(column)
        raise TransformError("no spare numeric column to put on a second axis")
