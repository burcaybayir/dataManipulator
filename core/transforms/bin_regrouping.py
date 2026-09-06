"""Bin regrouping — fold the long tail of a categorical column into one bar."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import Aggregation, ChartSpec
from core.transforms.base import Transform, TransformError
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class BinRegrouping(Transform):
    """Merge categories, either by an explicit mapping or by keeping the top N.

    Args:
        column: Categorical column to regroup. Defaults to the spec's series
            column, or its x column when there is no series.
        keep_top: Number of largest categories to keep; the rest are merged.
            Ignored when ``mapping`` is given.
        mapping: Explicit ``(from, to)`` pairs, applied before ``keep_top``.
        other_label: Name for the merged bin.
    """

    column: str | None = None
    keep_top: int | None = 3
    mapping: tuple[tuple[str, str], ...] = ()
    other_label: str = "Other"

    name = "bin_regrouping"
    label = "Regrouped bins"
    explanation = "Merges categories together, typically folding small ones into an 'Other' bin."
    why_misleading = (
        "Where the boundaries fall decides the ranking. Merging a competitor's "
        "segments, or scattering your own across several bins, changes who "
        "appears to lead without changing any total."
    )
    legitimate_when = (
        "A long tail genuinely obscures the chart and the merged bin is "
        "labeled with what it contains."
    )
    modifies_data = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", tuple(tuple(pair) for pair in self.mapping))
        if any(len(pair) != 2 for pair in self.mapping):
            raise TransformError("mapping entries must be (from, to) pairs")
        if not self.mapping and (self.keep_top is None or self.keep_top < 1):
            raise TransformError("keep_top must be at least 1 when no mapping is given")

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        column = self.column or spec.series or spec.x
        if column not in frame.columns:
            raise TransformError(f"column {column!r} is not in the data")
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_datetime64_any_dtype(
            frame[column]
        ):
            raise TransformError(f"{column!r} is not categorical, so its bins cannot be regrouped")

        out = frame.copy()
        labels = out[column].astype(str)
        if self.mapping:
            labels = labels.replace(dict(self.mapping))
        else:
            keepers = self._largest(out, spec, column)
            labels = labels.where(labels.isin(keepers), self.other_label)
        out[column] = labels
        return out, self._stamp(spec)

    def _largest(self, frame: pd.DataFrame, spec: ChartSpec, column: str) -> set[str]:
        """The ``keep_top`` categories with the largest totals."""
        labels = frame[column].astype(str)
        if spec.y is not None and spec.aggregation is not Aggregation.COUNT:
            totals = frame.groupby(labels, observed=True)[spec.y].sum()
        else:
            totals = labels.value_counts()
        keep_top = self.keep_top or 1
        return set(totals.sort_values(ascending=False).head(keep_top).index.astype(str))
