"""Inverted axis — flip the y direction so growth reads as decline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class InvertedAxis(Transform):
    """Reverse the y axis so larger values sit lower on the chart."""

    name = "inverted_axis"
    label = "Inverted axis"
    explanation = "Flips the y axis so that larger values are drawn further down."
    why_misleading = (
        "Readers assume up means more. An inverted axis turns a rising line "
        "into a falling one without changing a single number."
    )
    legitimate_when = (
        "The quantity is conventionally read downward — a golf score, a rank "
        "where 1 is best, depth below sea level."
    )

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        return frame, self._stamp(spec, invert_y=not spec.invert_y)
