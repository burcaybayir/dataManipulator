"""Aspect ratio — stretch or squash the plot to steepen or flatten a trend."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.spec import ChartSpec
from core.transforms.base import Transform, TransformError
from core.transforms.registry import register


@register
@dataclass(frozen=True, slots=True)
class AspectRatioChange(Transform):
    """Change the plot's width-to-height ratio.

    Args:
        width: Relative plot width.
        height: Relative plot height. A tall, narrow frame steepens a trend;
            a wide, short one flattens it.
    """

    width: float = 6.0
    height: float = 9.0

    name = "aspect_ratio"
    label = "Aspect ratio"
    explanation = "Changes the plot's width-to-height ratio without touching the data."
    why_misleading = (
        "A trend's apparent steepness is the angle of the line, and the angle "
        "is set by the frame. The same series can look like a cliff or a "
        "plateau depending only on the shape of the box it is drawn in."
    )
    legitimate_when = (
        "The chart is being fitted to a layout, and the aspect chosen makes "
        "the slope of interest roughly 45 degrees — the banking-to-45 rule."
    )

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise TransformError("width and height must both be positive")

    def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
        return frame, self._stamp(spec, aspect_ratio=(float(self.width), float(self.height)))

    @property
    def steepening(self) -> bool:
        """Whether this ratio makes trends look steeper than the default."""
        return self.width / self.height < 16.0 / 9.0
