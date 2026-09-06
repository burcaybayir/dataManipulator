"""Rebase to an index — set every series to 100 at a chosen point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.prepare import X
from core.spec import ChartSpec, ValueMode
from core.transforms.base import Transform, prepared_or_error
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class RebaseIndex(Transform):
    """Rescale each series so that its value at ``anchor`` equals 100.

    Args:
        anchor: The x value to rebase on. ``None`` uses the first point.
    """

    anchor: Any = None

    name = "rebase_index"
    label = "Rebased index"
    explanation = "Divides every series by its value at one x point and multiplies by 100."
    why_misleading = (
        "The choice of anchor is the whole result: rebasing on a series' worst "
        "point makes its recovery look strongest, and the absolute sizes of "
        "the series disappear entirely."
    )
    legitimate_when = (
        "Comparing the shape of series measured in different units or at very "
        "different scales, with the anchor stated on the chart."
    )

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        anchor = self.anchor if self.anchor is not None else self._first_x(frame, spec)
        return frame, self._stamp(spec, value_mode=ValueMode.INDEXED, index_anchor=anchor)

    def _first_x(self, frame: pd.DataFrame, spec: ChartSpec) -> Any:
        first = prepared_or_error(frame, spec).frame[X].iloc[0]
        return first.isoformat() if isinstance(first, pd.Timestamp) else first
