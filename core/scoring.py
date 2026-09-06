"""Impact scoring — how much a variant changes the reader's impression (FR-4).

Four numbers, computed by comparing a variant against the baseline it came
from:

``direction_flip``
    Does the naive reading of the chart reverse?
``slope_ratio``
    How much steeper the line is *as drawn*, including the axis range and the
    aspect ratio.
``magnitude_ratio``
    How much larger the change looks as a share of the plot height, ignoring
    the frame's shape.
``data_fidelity``
    What share of the baseline's rows the variant still stands on.

These are heuristics, not measurements of belief: they model a reader who
looks at the endpoints of a line and the height of the frame, which is what
a reader skimming a chart actually does. They are deliberately blunt, and the
UI labels them as estimates. Anything that depends on knowing what the reader
already believed is out of scope.

Two known blind spots follow from reading each series between its endpoints:

Shifts in level
    Swapping mean for median on a skewed distribution can move the whole line
    without changing its slope. A reader who reads the axis labels rather than
    the shape is misled by exactly that, and these scores stay near 1.

The second axis
    ``dual_axis`` rescales a series relative to its neighbours, which lives in
    the renderer rather than in the prepared values, so it scores as no change
    at all until Stage 2 exists.

Both are pinned by tests, so that a metric which starts catching one fails
loudly rather than silently changing what the scores mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from core.prepare import SERIES, VALUE, PreparedChart, prepare
from core.spec import ChartSpec
from core.transforms.stack import TransformStack

#: Visual slopes below this are treated as flat. Slopes are expressed as a
#: fraction of plot height over plot width, so this is a hundredth of a percent
#: of the frame — far below what any reader could see.
FLAT_TOLERANCE = 1e-4

#: Ratio reported when the baseline is flat and the variant is not: the change
#: is not a multiple of anything, it is created from nothing.
UNBOUNDED = math.inf


class Direction(StrEnum):
    """The story a reader takes from the shape of the line."""

    RISING = "rising"
    FALLING = "falling"
    FLAT = "flat"


@dataclass(frozen=True, slots=True)
class ImpactScores:
    """How differently a variant reads from its baseline.

    Attributes:
        baseline_direction: Which way the baseline appears to go.
        variant_direction: Which way the variant appears to go.
        slope_ratio: Variant visual slope over baseline visual slope. Negative
            when the direction reverses; ``inf`` when the baseline was flat.
        magnitude_ratio: Apparent effect size over the true effect size, both
            as a share of the plot height.
        data_fidelity: Share of the baseline's rows still represented.
        baseline_slope: Baseline visual slope, in plot heights per plot width.
        variant_slope: The same for the variant.
    """

    baseline_direction: Direction
    variant_direction: Direction
    slope_ratio: float
    magnitude_ratio: float
    data_fidelity: float
    baseline_slope: float
    variant_slope: float

    #: These numbers estimate an impression, not a measurement. Surface this
    #: wherever they are shown.
    is_heuristic = True

    @property
    def direction_flip(self) -> bool:
        """Whether the naive reading of the chart reverses.

        A baseline that reads flat cannot flip — there was no direction to
        reverse — so inventing a trend from flat data shows up in
        :attr:`slope_ratio` instead.
        """
        if Direction.FLAT in (self.baseline_direction, self.variant_direction):
            return False
        return self.baseline_direction is not self.variant_direction

    @property
    def dropped_fraction(self) -> float:
        """Share of the baseline's rows the variant leaves out."""
        return max(0.0, 1.0 - self.data_fidelity)

    @property
    def exaggerates(self) -> bool:
        """Whether the change looks larger than it is."""
        return self.magnitude_ratio > 1.0

    def summary(self) -> str:
        """One line a caption can use, phrased as an estimate."""
        parts: list[str] = []
        if self.direction_flip:
            parts.append(
                f"reverses the trend ({self.baseline_direction} becomes {self.variant_direction})"
            )
        elif self.baseline_direction is Direction.FLAT and self.variant_direction is not (
            Direction.FLAT
        ):
            parts.append(f"invents a {self.variant_direction} trend from flat data")

        if math.isinf(self.slope_ratio):
            parts.append("steepens a line that had no slope")
        elif abs(self.slope_ratio) >= 1.1:
            parts.append(f"looks {abs(self.slope_ratio):.1f}x steeper")
        elif abs(self.slope_ratio) <= 0.9:
            parts.append(f"looks {1 / max(abs(self.slope_ratio), 1e-9):.1f}x flatter")

        if self.magnitude_ratio >= 1.1:
            parts.append(f"the change fills {self.magnitude_ratio:.1f}x more of the frame")
        if self.dropped_fraction > 0:
            parts.append(f"drops {self.dropped_fraction:.0%} of the rows")

        return "; ".join(parts) if parts else "reads much like the baseline"


def score(baseline: PreparedChart, variant: PreparedChart) -> ImpactScores:
    """Compare a prepared variant against a prepared baseline.

    Both must come from the same source data; nothing here checks that, since
    the caller is the one holding the manifest.
    """
    baseline_slope = visual_slope(baseline)
    variant_slope = visual_slope(variant)

    return ImpactScores(
        baseline_direction=_direction(baseline_slope),
        variant_direction=_direction(variant_slope),
        slope_ratio=_ratio(variant_slope, baseline_slope),
        magnitude_ratio=_ratio(_frame_share(variant), _frame_share(baseline)),
        data_fidelity=_fidelity(baseline, variant),
        baseline_slope=baseline_slope,
        variant_slope=variant_slope,
    )


def score_stack(
    frame: pd.DataFrame, spec: ChartSpec, stack: TransformStack
) -> tuple[PreparedChart, ImpactScores]:
    """Apply ``stack`` to ``frame`` and score the result against the baseline.

    Returns:
        The prepared variant and its scores.
    """
    baseline = prepare(frame, spec)
    variant = prepare(*stack.apply(frame, spec))
    return variant, score(baseline, variant)


def visual_slope(prepared: PreparedChart) -> float:
    """The slope of the drawn line, in plot heights per plot width.

    This is the number the reader actually responds to. It moves when the data
    moves, but also when the axis is truncated (the same change fills more of
    the frame), when the window narrows (the same rise is stretched across the
    full width), when the frame is reshaped, and when the axis is inverted.

    On a multi-series chart it is the average of the series' slopes, all
    measured against the shared y axis — the reader sees the individual lines,
    not their sum, and series moving in opposite directions genuinely do read
    as no clear overall trend.
    """
    span = prepared.y_domain[1] - prepared.y_domain[0]
    if span <= 0:
        return 0.0

    changes = list(series_changes(prepared).values())
    if not changes:
        return 0.0

    width, height = prepared.spec.aspect_ratio
    slope = (sum(changes) / len(changes) / span) * (height / width)
    return -slope if prepared.spec.invert_y else slope


def series_changes(prepared: PreparedChart) -> dict[str, float]:
    """First-to-last change for each plotted series, in plotted order.

    Series with fewer than two points contribute nothing to read a trend from
    and are left out.
    """
    changes: dict[str, float] = {}
    for name, group in prepared.frame.groupby(SERIES, sort=False, observed=True):
        values = group[VALUE].dropna()
        if len(values) >= 2:
            changes[str(name)] = float(values.iloc[-1] - values.iloc[0])
    return changes


def _frame_share(prepared: PreparedChart) -> float:
    """How much of the plot height the average series' change fills.

    Magnitudes are taken unsigned, so series moving in opposite directions add
    up rather than cancelling: the chart still shows large movement, whatever
    a reader concludes about its direction.
    """
    span = prepared.y_domain[1] - prepared.y_domain[0]
    if span <= 0:
        return 0.0
    changes = [abs(v) for v in series_changes(prepared).values()]
    if not changes:
        return 0.0
    return sum(changes) / len(changes) / span


def _direction(slope: float) -> Direction:
    if abs(slope) <= FLAT_TOLERANCE:
        return Direction.FLAT
    return Direction.RISING if slope > 0 else Direction.FALLING


def _ratio(variant: float, baseline: float) -> float:
    """Variant over baseline, with the degenerate cases pinned down.

    A flat baseline has no magnitude to be a multiple of, so a variant that
    moves scores :data:`UNBOUNDED` rather than dividing by something near zero.
    """
    if abs(baseline) <= FLAT_TOLERANCE:
        return 1.0 if abs(variant) <= FLAT_TOLERANCE else UNBOUNDED
    return variant / baseline


def _fidelity(baseline: PreparedChart, variant: PreparedChart) -> float:
    """Share of the baseline's rows the variant still stands on."""
    represented = baseline.represented_rows
    if represented == 0:
        return 1.0
    return min(variant.represented_rows / represented, 1.0)
