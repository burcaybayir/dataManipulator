"""Cumulative totals — plot the running sum instead of the period value."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class Cumulative(Transform):
    """Plot each series as a running total."""

    name = "cumulative"
    label = "Cumulative total"
    explanation = "Plots the running sum of the series instead of each period's value."
    why_misleading = (
        "A cumulative chart of any non-negative quantity only ever rises, so "
        "a business in decline still produces a reassuring upward curve."
    )
    legitimate_when = (
        "The total to date is the quantity of interest — signups against a "
        "target, rainfall for the season."
    )

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        return frame, self._stamp(spec, cumulative=True)
