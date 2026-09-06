"""Ratio vs absolute — restate values as percent change, or back again."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import ChartSpec, ValueMode
from core.transforms.base import Transform, TransformError
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class RatioVsAbsolute(Transform):
    """Switch between absolute values and period-over-period percent change.

    Args:
        mode: ``percent_change`` or ``absolute``.
    """

    mode: ValueMode = ValueMode.PERCENT_CHANGE

    name = "ratio_vs_absolute"
    label = "Ratio instead of absolute"
    explanation = "Plots period-over-period percent change rather than the underlying values."
    why_misleading = (
        "Percent change flatters small bases: three sales becoming six is "
        "'100% growth', while a large base posting a far bigger absolute gain "
        "looks stagnant."
    )
    legitimate_when = (
        "Comparing series of very different sizes, where the rate of change "
        "is the thing being compared and the bases are also shown."
    )

    def __post_init__(self) -> None:
        mode = ValueMode(self.mode)
        if mode is ValueMode.INDEXED:
            raise TransformError("use the rebase_index transform for indexed values")
        object.__setattr__(self, "mode", mode)

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        return frame, self._stamp(spec, value_mode=self.mode)
