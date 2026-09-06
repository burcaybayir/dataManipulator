"""Outlier drop — quietly remove the points that spoil the story."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform, TransformError
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class OutlierDrop(Transform):
    """Drop rows whose measure lies beyond ``z_threshold`` standard deviations.

    Args:
        z_threshold: Distance from the mean, in standard deviations, beyond
            which a row is removed.
        column: Measure column to judge. Defaults to the spec's y column.
    """

    z_threshold: float = 2.0
    column: str | None = None

    name = "outlier_drop"
    label = "Dropped outliers"
    explanation = "Removes rows whose measure is far from the mean, without noting the removal."
    why_misleading = (
        "Outliers are often the story — the outage, the record quarter, the "
        "fraud. Removing them undisclosed shrinks variance and makes whatever "
        "remains look like the normal state of affairs."
    )
    legitimate_when = (
        "The excluded points are known measurement errors, and the exclusion "
        "and its rule are stated alongside the chart."
    )
    modifies_data = True

    def __post_init__(self) -> None:
        if self.z_threshold <= 0:
            raise TransformError("z_threshold must be positive")

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        column = self.column or spec.y
        if column is None:
            raise TransformError("no measure column to judge outliers against")
        if column not in frame.columns:
            raise TransformError(f"column {column!r} is not in the data")
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TransformError(f"{column!r} is not numeric")

        values = frame[column].astype(float)
        spread = float(values.std(ddof=0))
        if not np.isfinite(spread) or spread == 0:
            return frame, self._stamp(spec)

        z_scores = (values - float(values.mean())).abs() / spread
        kept = frame[z_scores <= self.z_threshold]
        if kept.empty:
            raise TransformError("dropping outliers would remove every row")
        return kept.reset_index(drop=True), self._stamp(spec)
