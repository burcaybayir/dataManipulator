"""Aggregation swap — change how rows sharing an x value are collapsed."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import Aggregation, ChartSpec
from core.transforms.base import Transform, TransformError
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class AggregationSwap(Transform):
    """Switch the aggregation between sum, mean, median and count.

    Args:
        target: The aggregation to switch to.
    """

    target: Aggregation = Aggregation.MEAN

    name = "aggregation_swap"
    label = "Aggregation swap"
    explanation = "Collapses duplicate x values with a different statistic."
    why_misleading = (
        "Mean and median diverge exactly where the distribution is skewed. "
        "Picking whichever is higher, without saying which was used, changes "
        "the level without touching a number."
    )
    legitimate_when = (
        "The statistic matches the question — median for a typical case, mean "
        "for a total divided evenly — and the chart says which one it is."
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", Aggregation(self.target))

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        if self.target is not Aggregation.COUNT and spec.y is None:
            raise TransformError(f"{self.target.value} needs a measure column, but y is unset")
        return frame, self._stamp(spec, aggregation=self.target)
