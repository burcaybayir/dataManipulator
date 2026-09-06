"""Materialize a :class:`~core.spec.ChartSpec` into plot-ready values.

This is the data half of rendering, pulled ahead of the renderer because a
transform that changes nothing observable is not a transform. Everything a
distortion can do to the numbers happens here, in a fixed order:

``window -> aggregate -> smooth -> accumulate -> restate``

Each step is skipped unless the spec asks for it, so a baseline spec yields a
plain aggregation of the source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.spec import Aggregation, ChartSpec, ValueMode

#: Column names of the long-form frame every renderer and scorer consumes.
X = "x"
SERIES = "series"
VALUE = "value"

#: Label used when a spec has no series column.
SINGLE_SERIES = "all"


class PrepareError(Exception):
    """Raised when a spec cannot be applied to the data it names."""


@dataclass(frozen=True, slots=True)
class PreparedChart:
    """Plot-ready values plus the axis bounds a renderer would use.

    Attributes:
        frame: Long-form ``x``/``series``/``value`` rows, sorted by x.
        spec: The spec these values came from.
        y_domain: ``(low, high)`` the y axis would span.
        dropped_rows: Rows removed by the x window.
        source_rows: Rows in the frame this was prepared from, before the
            window was applied. Transforms that drop rows shrink it, so
            ``represented_rows`` measures how much of the original data a
            variant still stands on.
    """

    frame: pd.DataFrame
    spec: ChartSpec
    y_domain: tuple[float, float]
    dropped_rows: int = 0
    source_rows: int = 0

    @property
    def represented_rows(self) -> int:
        """Source rows still represented in the plotted values."""
        return max(self.source_rows - self.dropped_rows, 0)

    @property
    def series_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.frame[SERIES].tolist()))

    @property
    def values(self) -> pd.Series:
        return self.frame[VALUE]

    def series(self, name: str) -> pd.Series:
        """Values for one series, indexed by x."""
        subset = self.frame[self.frame[SERIES] == name]
        return pd.Series(subset[VALUE].to_numpy(), index=subset[X].to_numpy(), name=name)

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


def prepare(frame: pd.DataFrame, spec: ChartSpec) -> PreparedChart:
    """Apply ``spec`` to ``frame`` and return the values a chart would draw.

    Raises:
        PrepareError: The spec names a column the frame does not have, or the
            window it asks for excludes every row.
    """
    _check_columns(frame, spec)

    windowed, dropped = _apply_window(frame, spec)
    if windowed.empty:
        raise PrepareError("the selected x range contains no rows")

    long = _aggregate(windowed, spec)
    long = _smooth(long, spec)
    long = _accumulate(long, spec)
    long = _restate(long, spec)

    return PreparedChart(
        frame=long.reset_index(drop=True),
        spec=spec,
        y_domain=_y_domain(long[VALUE], spec),
        dropped_rows=dropped,
        source_rows=len(frame),
    )


def _check_columns(frame: pd.DataFrame, spec: ChartSpec) -> None:
    missing = [c for c in spec.columns if c not in frame.columns]
    if missing:
        raise PrepareError(f"columns not in the data: {', '.join(missing)}")
    if spec.y is not None and not pd.api.types.is_numeric_dtype(frame[spec.y]):
        raise PrepareError(f"{spec.y!r} is not numeric, so it cannot be a measure")


def _apply_window(frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, int]:
    """Restrict rows to ``spec.x_range``, inclusive at both ends."""
    if spec.x_range is None:
        return frame, 0
    start, end = spec.x_range
    column = frame[spec.x]

    if pd.api.types.is_datetime64_any_dtype(column):
        lower = pd.to_datetime(start) if start is not None else None
        upper = pd.to_datetime(end) if end is not None else None
    elif pd.api.types.is_numeric_dtype(column):
        lower, upper = start, end
    else:
        return _apply_categorical_window(frame, spec, start, end)

    mask = pd.Series(True, index=frame.index)
    if lower is not None:
        mask &= column >= lower
    if upper is not None:
        mask &= column <= upper
    return frame[mask], int((~mask).sum())


def _apply_categorical_window(
    frame: pd.DataFrame, spec: ChartSpec, start: Any, end: Any
) -> tuple[pd.DataFrame, int]:
    """Window a non-ordered x axis by position in first-appearance order."""
    order = list(dict.fromkeys(frame[spec.x].tolist()))
    try:
        first = order.index(start) if start is not None else 0
        last = order.index(end) if end is not None else len(order) - 1
    except ValueError as exc:
        raise PrepareError(f"{exc.args[0]} is not a value of {spec.x!r}") from exc
    keep = set(order[first : last + 1])
    mask = frame[spec.x].isin(keep)
    return frame[mask], int((~mask).sum())


def _aggregate(frame: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    """Collapse rows sharing an x (and series) value into one point."""
    keys = [spec.x] + ([spec.series] if spec.series else [])

    if spec.aggregation is Aggregation.COUNT:
        grouped = frame.groupby(keys, dropna=False, observed=True).size().reset_index(name=VALUE)
    else:
        measure = spec.y
        assert measure is not None  # guaranteed by ChartSpec validation
        grouped = (
            frame.groupby(keys, dropna=False, observed=True)[measure]
            .agg(spec.aggregation.value)
            .reset_index(name=VALUE)
        )

    grouped = grouped.rename(columns={spec.x: X})
    if spec.series:
        grouped = grouped.rename(columns={spec.series: SERIES})
        grouped[SERIES] = grouped[SERIES].astype(str)
    else:
        grouped[SERIES] = SINGLE_SERIES

    grouped[VALUE] = pd.to_numeric(grouped[VALUE], errors="coerce").astype(float)
    return grouped[[X, SERIES, VALUE]].sort_values([SERIES, X], kind="stable")


def _per_series(long: pd.DataFrame, fn: Any) -> pd.DataFrame:
    """Apply ``fn`` to each series' value column independently."""
    out = long.copy()
    out[VALUE] = out.groupby(SERIES, observed=True)[VALUE].transform(fn)
    return out


def _smooth(long: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    """Replace each point with a trailing rolling mean.

    ``min_periods=1`` keeps the leading points rather than blanking them,
    which is what a chart tool doing this quietly would do.
    """
    if spec.smoothing_window is None:
        return long
    window = spec.smoothing_window
    return _per_series(long, lambda s: s.rolling(window=window, min_periods=1).mean())


def _accumulate(long: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    if not spec.cumulative:
        return long
    return _per_series(long, lambda s: s.cumsum())


def _restate(long: pd.DataFrame, spec: ChartSpec) -> pd.DataFrame:
    """Convert absolute values to percent change or an index."""
    match spec.value_mode:
        case ValueMode.ABSOLUTE:
            return long
        case ValueMode.PERCENT_CHANGE:
            return _per_series(long, lambda s: s.pct_change().mul(100.0).fillna(0.0))
        case ValueMode.INDEXED:
            return _rebase(long, spec.index_anchor)
    raise PrepareError(f"unsupported value mode: {spec.value_mode}")


def _rebase(long: pd.DataFrame, anchor: Any) -> pd.DataFrame:
    """Rescale each series so its value at ``anchor`` is 100."""
    out = long.copy()
    frames = []
    for name, group in out.groupby(SERIES, observed=True, sort=False):
        matched = group[group[X] == anchor]
        if matched.empty:
            raise PrepareError(f"index anchor {anchor!r} is not present in series {name!r}")
        base = float(matched[VALUE].iloc[0])
        if base == 0:
            raise PrepareError(f"cannot rebase series {name!r} on a zero value")
        group = group.copy()
        group[VALUE] = group[VALUE] / base * 100.0
        frames.append(group)
    return pd.concat(frames)


def _y_domain(values: pd.Series, spec: ChartSpec) -> tuple[float, float]:
    """The axis bounds a renderer would use for these values.

    The honest default lives here: every family is zero-based unless the spec
    says otherwise, so that raising the floor is always a visible, attributable
    change rather than something the baseline did quietly. Line and scatter
    charts may legitimately start above zero — that is what the truncated_axis
    transform is for, applied deliberately and labeled.
    """
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    data_low = float(finite.min()) if not finite.empty else 0.0
    data_high = float(finite.max()) if not finite.empty else 0.0

    low = spec.y_min if spec.y_min is not None else min(0.0, data_low)
    high = spec.y_max if spec.y_max is not None else data_high
    if high <= low:
        high = low + abs(low or 1.0) * 0.1
    return (low, high)
