"""The chart specification — the object transforms rewrite.

A :class:`ChartSpec` is a complete, declarative description of a chart: which
columns to plot, how to aggregate them, and every visual knob a distortion
might reach for. It is frozen, so a transform produces a new spec rather than
mutating the one it was handed, which is what makes a transform stack
replayable from a manifest.

Defaults are deliberately the *honest* ones: zero-based axis, full range,
absolute values, no smoothing. A baseline chart is simply a spec nobody has
transformed yet.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self


class ChartFamily(StrEnum):
    """Supported chart types (FR-2)."""

    LINE = "line"
    BAR = "bar"
    AREA = "area"
    SCATTER = "scatter"
    PIE = "pie"


class Aggregation(StrEnum):
    """How rows sharing an x value are collapsed."""

    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    COUNT = "count"


class ValueMode(StrEnum):
    """Whether the y axis shows raw values or a derived ratio."""

    ABSOLUTE = "absolute"
    PERCENT_CHANGE = "percent_change"
    INDEXED = "indexed"


#: Families where a non-zero-based y axis is always a distortion, because the
#: mark's length (not just its position) encodes the value.
LENGTH_ENCODED_FAMILIES = frozenset({ChartFamily.BAR, ChartFamily.AREA, ChartFamily.PIE})

DEFAULT_ASPECT_RATIO = (16.0, 9.0)


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """A declarative chart description.

    Attributes:
        family: Chart type to render.
        x: Column mapped to the x axis.
        y: Column mapped to the y axis. ``None`` is only valid with
            :attr:`Aggregation.COUNT`, which counts rows per x value.
        series: Optional column that splits the data into multiple series.
        aggregation: How duplicate x values are collapsed.
        value_mode: Absolute values, percent change, or rebased index.
        y_min: Explicit y-axis floor. ``None`` means "let the renderer choose",
            which for length-encoded families means zero.
        y_max: Explicit y-axis ceiling, or ``None`` for the data maximum.
        invert_y: Flip the y axis direction.
        x_range: Inclusive ``(start, end)`` window applied to the x column, or
            ``None`` for the full range.
        smoothing_window: Rolling-mean window in points; ``None`` for raw data.
        cumulative: Plot the running total instead of the period value.
        index_anchor: x value rebased to 100 when ``value_mode`` is INDEXED.
        aspect_ratio: ``(width, height)`` used to steepen or flatten a trend.
        secondary_y: Series values rendered against an independent y axis.
        title: Optional chart title.
        notes: Free-form provenance notes carried into exports.
    """

    family: ChartFamily
    x: str
    y: str | None = None
    series: str | None = None
    aggregation: Aggregation = Aggregation.SUM
    value_mode: ValueMode = ValueMode.ABSOLUTE
    y_min: float | None = None
    y_max: float | None = None
    invert_y: bool = False
    x_range: tuple[Any, Any] | None = None
    smoothing_window: int | None = None
    cumulative: bool = False
    index_anchor: Any | None = None
    aspect_ratio: tuple[float, float] = DEFAULT_ASPECT_RATIO
    secondary_y: tuple[str, ...] = ()
    title: str | None = None
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.y is None and self.aggregation is not Aggregation.COUNT:
            raise ValueError("y may only be omitted when aggregation is COUNT")
        if self.smoothing_window is not None and self.smoothing_window < 2:
            raise ValueError("smoothing_window must be at least 2 points")
        if self.y_min is not None and self.y_max is not None and self.y_min >= self.y_max:
            raise ValueError("y_min must be below y_max")
        if self.value_mode is ValueMode.INDEXED and self.index_anchor is None:
            raise ValueError("INDEXED value_mode requires an index_anchor")
        width, height = self.aspect_ratio
        if width <= 0 or height <= 0:
            raise ValueError("aspect_ratio components must be positive")

    def evolve(self, **changes: Any) -> Self:
        """Return a copy with ``changes`` applied.

        Transforms use this instead of mutating, so the original spec stays
        available for the side-by-side comparison against the baseline.
        """
        return dataclasses.replace(self, **changes)

    @property
    def columns(self) -> tuple[str, ...]:
        """Every dataset column this spec references, in mapping order."""
        names = [self.x]
        if self.y is not None:
            names.append(self.y)
        if self.series is not None:
            names.append(self.series)
        return tuple(dict.fromkeys(names))

    @property
    def is_baseline(self) -> bool:
        """True when no distortion knob has been touched.

        This is a structural check, not a claim of honesty: a baseline spec can
        still be built on cherry-picked *data*. It answers "has a transform
        rewritten this spec?", which is what the gallery needs to know.
        """
        return (
            self.y_min is None
            and self.y_max is None
            and not self.invert_y
            and self.x_range is None
            and self.smoothing_window is None
            and not self.cumulative
            and self.value_mode is ValueMode.ABSOLUTE
            and self.aspect_ratio == DEFAULT_ASPECT_RATIO
            and not self.secondary_y
        )

    @property
    def requires_zero_baseline(self) -> bool:
        """Whether truncating this chart's y axis is indefensible.

        Every family is drawn zero-based by default. In these ones the mark's
        *length* encodes the value, so a raised floor misstates magnitudes
        outright rather than merely exaggerating a trend.
        """
        return self.family in LENGTH_ENCODED_FAMILIES

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the reproducibility manifest (FR-8)."""
        raw = dataclasses.asdict(self)
        raw["family"] = self.family.value
        raw["aggregation"] = self.aggregation.value
        raw["value_mode"] = self.value_mode.value
        raw["aspect_ratio"] = list(self.aspect_ratio)
        raw["secondary_y"] = list(self.secondary_y)
        raw["notes"] = list(self.notes)
        if self.x_range is not None:
            raw["x_range"] = list(self.x_range)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChartSpec:
        """Rebuild a spec produced by :meth:`to_dict`."""
        data = dict(raw)
        data["family"] = ChartFamily(data["family"])
        data["aggregation"] = Aggregation(data["aggregation"])
        data["value_mode"] = ValueMode(data.get("value_mode", ValueMode.ABSOLUTE))
        data["aspect_ratio"] = tuple(data.get("aspect_ratio", DEFAULT_ASPECT_RATIO))
        data["secondary_y"] = tuple(data.get("secondary_y", ()))
        data["notes"] = tuple(data.get("notes", ()))
        if data.get("x_range") is not None:
            data["x_range"] = tuple(data["x_range"])
        return cls(**data)
