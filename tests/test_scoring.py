"""Calibration tests for impact scoring.

The expected numbers here are computed by hand from the definitions in
``core.scoring``, not read off a previous run. Visual slope is
``(change / y-span) * (height / width)``, so for the fixture below the
baseline slope is ``(40 / 140) * (9 / 16) = 0.1607``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.prepare import PreparedChart, prepare
from core.scoring import (
    FLAT_TOLERANCE,
    UNBOUNDED,
    Direction,
    ImpactScores,
    score,
    score_stack,
    series_changes,
    visual_slope,
)
from core.spec import Aggregation, ChartFamily, ChartSpec
from core.transforms import (
    AggregationSwap,
    AspectRatioChange,
    CherryPickedWindow,
    Cumulative,
    DualAxis,
    InvertedAxis,
    OutlierDrop,
    Smoothing,
    Transform,
    TransformStack,
    TruncatedAxis,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Five months rising 100 -> 140, with a dip to 90 in the middle."""
    return pd.DataFrame(
        {
            "m": pd.to_datetime([f"2024-0{i}-01" for i in range(1, 6)]),
            "v": [100.0, 120.0, 90.0, 130.0, 140.0],
        }
    )


@pytest.fixture
def spec() -> ChartSpec:
    return ChartSpec(family=ChartFamily.LINE, x="m", y="v")


def scores_for(frame: pd.DataFrame, spec: ChartSpec, *transforms: Transform) -> ImpactScores:
    return score_stack(frame, spec, TransformStack().push(*transforms))[1]


# --- the baseline against itself ---------------------------------------------


def test_the_baseline_slope_is_the_documented_formula(frame: pd.DataFrame, spec: ChartSpec) -> None:
    prepared = prepare(frame, spec)
    assert prepared.y_domain == (0.0, 140.0)
    assert visual_slope(prepared) == pytest.approx((40 / 140) * (9 / 16))


def test_an_empty_stack_scores_as_no_change(frame: pd.DataFrame, spec: ChartSpec) -> None:
    result = scores_for(frame, spec)
    assert result.slope_ratio == pytest.approx(1.0)
    assert result.magnitude_ratio == pytest.approx(1.0)
    assert result.data_fidelity == 1.0
    assert not result.direction_flip
    assert result.summary() == "reads much like the baseline"


# --- calibration against hand-computed values --------------------------------


def test_truncating_the_axis_multiplies_slope_and_magnitude_alike(
    frame: pd.DataFrame, spec: ChartSpec
) -> None:
    # Data spans 90..140, so a 5% headroom floor sits at 87.5 and the y span
    # falls from 140 to 52.5: 140 / 52.5 = 2.667.
    result = scores_for(frame, spec, TruncatedAxis(headroom=0.05))
    assert result.slope_ratio == pytest.approx(140 / 52.5)
    assert result.magnitude_ratio == pytest.approx(140 / 52.5)
    assert result.data_fidelity == 1.0
    assert result.exaggerates
    assert result.summary() == ("looks 2.7x steeper; the change fills 2.7x more of the frame")


def test_inverting_the_axis_flips_direction_without_changing_magnitude(
    frame: pd.DataFrame, spec: ChartSpec
) -> None:
    result = scores_for(frame, spec, InvertedAxis())
    assert result.slope_ratio == pytest.approx(-1.0)
    assert result.magnitude_ratio == pytest.approx(1.0)
    assert result.direction_flip
    assert result.baseline_direction is Direction.RISING
    assert result.variant_direction is Direction.FALLING
    assert result.summary() == "reverses the trend (rising becomes falling)"


def test_reshaping_the_frame_steepens_the_line_but_not_the_change(
    frame: pd.DataFrame, spec: ChartSpec
) -> None:
    # A 6:9 frame against the 16:9 default: (9/6) / (9/16) = 2.667.
    result = scores_for(frame, spec, AspectRatioChange(width=6.0, height=9.0))
    assert result.slope_ratio == pytest.approx((16 / 9) / (6 / 9))
    assert result.magnitude_ratio == pytest.approx(1.0)


def test_a_wide_frame_flattens_the_line(frame: pd.DataFrame, spec: ChartSpec) -> None:
    result = scores_for(frame, spec, AspectRatioChange(width=32.0, height=9.0))
    assert result.slope_ratio == pytest.approx(0.5)


def test_cumulative_totals_exaggerate_the_climb(frame: pd.DataFrame, spec: ChartSpec) -> None:
    # Running totals end at 580 having started at 100: (480/580) / (40/140).
    result = scores_for(frame, spec, Cumulative())
    assert result.slope_ratio == pytest.approx((480 / 580) / (40 / 140))


def test_smoothing_flattens_the_line(frame: pd.DataFrame, spec: ChartSpec) -> None:
    result = scores_for(frame, spec, Smoothing(window=3))
    assert result.slope_ratio < 1.0
    assert not result.exaggerates
    assert "flatter" in result.summary()


def test_dropping_outliers_costs_fidelity(frame: pd.DataFrame, spec: ChartSpec) -> None:
    result = scores_for(frame, spec, OutlierDrop(z_threshold=1.0))
    assert result.data_fidelity == pytest.approx(3 / 5)
    assert result.dropped_fraction == pytest.approx(2 / 5)
    assert "drops 40% of the rows" in result.summary()


def test_a_window_costs_fidelity_and_steepens(frame: pd.DataFrame, spec: ChartSpec) -> None:
    result = scores_for(frame, spec, CherryPickedWindow(start="2024-03-01"))
    assert result.data_fidelity == pytest.approx(3 / 5)
    assert result.slope_ratio > 1.0


def test_stacked_transforms_compound(frame: pd.DataFrame, spec: ChartSpec) -> None:
    single = scores_for(frame, spec, TruncatedAxis(headroom=0.05))
    stacked = scores_for(
        frame, spec, TruncatedAxis(headroom=0.05), AspectRatioChange(width=6.0, height=9.0)
    )
    assert stacked.slope_ratio == pytest.approx(single.slope_ratio * (16 / 9) / (6 / 9))


# --- direction and the flat cases --------------------------------------------


def test_a_falling_series_reads_as_falling() -> None:
    frame = pd.DataFrame({"m": ["a", "b"], "v": [100.0, 50.0]})
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v")
    assert scores_for(frame, spec).baseline_direction is Direction.FALLING


def test_a_flat_series_reads_as_flat() -> None:
    frame = pd.DataFrame({"m": ["a", "b"], "v": [100.0, 100.0]})
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v")
    result = scores_for(frame, spec)
    assert result.baseline_direction is Direction.FLAT
    assert not result.direction_flip


def test_inventing_a_trend_from_flat_data_scores_as_unbounded() -> None:
    frame = pd.DataFrame({"m": ["a", "b", "c"], "v": [100.0, 100.0, 100.0]})
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v")
    result = scores_for(frame, spec, Cumulative())
    assert result.baseline_direction is Direction.FLAT
    assert result.variant_direction is Direction.RISING
    assert math.isinf(result.slope_ratio)
    assert not result.direction_flip  # nothing was reversed; a trend was created
    assert "invents a rising trend" in result.summary()
    assert "steepens a line that had no slope" in result.summary()


def test_a_flat_variant_of_a_flat_baseline_scores_as_unchanged() -> None:
    flat = pd.DataFrame({"m": ["a", "b"], "v": [7.0, 7.0]})
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v")
    result = scores_for(flat, spec, AspectRatioChange(width=2.0, height=9.0))
    assert result.slope_ratio == 1.0
    assert result.magnitude_ratio == 1.0


def test_a_single_point_has_no_slope() -> None:
    frame = pd.DataFrame({"m": ["a"], "v": [5.0]})
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v")
    prepared = prepare(frame, spec)
    assert visual_slope(prepared) == 0.0
    assert scores_for(frame, spec).baseline_direction is Direction.FLAT


def test_the_flat_tolerance_is_below_visible_resolution() -> None:
    assert FLAT_TOLERANCE < 1e-3
    assert math.isinf(UNBOUNDED)


# --- multi-series and fidelity edges -----------------------------------------


def test_scores_average_the_series_a_reader_sees() -> None:
    frame = pd.DataFrame(
        {
            "m": ["a", "a", "b", "b"],
            "v": [10.0, 30.0, 20.0, 60.0],
            "r": ["n", "s", "n", "s"],
        }
    )
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v", series="r")
    assert series_changes(prepare(frame, spec)) == {"n": 10.0, "s": 30.0}
    assert scores_for(frame, spec).baseline_direction is Direction.RISING


def test_series_pulling_in_opposite_directions_read_as_no_clear_trend() -> None:
    frame = pd.DataFrame(
        {
            "m": ["a", "a", "b", "b"],
            "v": [10.0, 30.0, 30.0, 10.0],
            "r": ["n", "s", "n", "s"],
        }
    )
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v", series="r")
    result = scores_for(frame, spec)
    assert result.baseline_direction is Direction.FLAT
    # The chart is not motionless, though: magnitude counts both moves.
    assert prepare(frame, spec).y_domain == (0.0, 30.0)


def test_a_series_too_short_to_have_a_trend_is_left_out() -> None:
    frame = pd.DataFrame(
        {
            "m": ["a", "a", "b"],
            "v": [10.0, 30.0, 20.0],
            "r": ["n", "s", "n"],
        }
    )
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v", series="r")
    assert set(series_changes(prepare(frame, spec))) == {"n"}


def test_fidelity_never_exceeds_one(frame: pd.DataFrame, spec: ChartSpec) -> None:
    baseline = prepare(frame, spec)
    bigger = prepare(pd.concat([frame, frame], ignore_index=True), spec)
    assert score(baseline, bigger).data_fidelity == 1.0


def test_fidelity_of_an_empty_baseline_is_one(frame: pd.DataFrame, spec: ChartSpec) -> None:
    prepared = prepare(frame, spec)
    empty = PreparedChart(
        frame=prepared.frame, spec=spec, y_domain=(0.0, 1.0), dropped_rows=0, source_rows=0
    )
    assert score(empty, prepared).data_fidelity == 1.0


def test_a_zero_height_frame_has_no_slope(frame: pd.DataFrame, spec: ChartSpec) -> None:
    prepared = prepare(frame, spec)
    degenerate = PreparedChart(
        frame=prepared.frame, spec=spec, y_domain=(5.0, 5.0), source_rows=prepared.source_rows
    )
    assert visual_slope(degenerate) == 0.0
    assert score(degenerate, degenerate).magnitude_ratio == 1.0


def test_scores_are_labeled_as_estimates() -> None:
    assert ImpactScores.is_heuristic


# --- known blind spots -------------------------------------------------------
#
# These pin limitations documented in core.scoring. If a change starts catching
# one, the test fails and says so rather than quietly redefining the scores.


def test_a_shift_in_level_is_not_caught() -> None:
    """Mean to median on a skewed distribution moves the line, not its slope."""
    frame = pd.DataFrame(
        {
            "m": ["a", "a", "a", "b", "b", "b"],
            "v": [1.0, 2.0, 60.0, 2.0, 4.0, 120.0],
        }
    )
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v", aggregation=Aggregation.MEAN)
    variant, result = score_stack(
        frame, spec, TransformStack().push(AggregationSwap(target=Aggregation.MEDIAN))
    )

    baseline_values = prepare(frame, spec).values.tolist()
    assert baseline_values == [21.0, 42.0]
    assert variant.values.tolist() == [2.0, 4.0]  # the level collapses
    assert result.slope_ratio == pytest.approx(1.0)  # the scores do not see it
    assert result.summary() == "reads much like the baseline"


def test_a_second_axis_is_not_caught(frame: pd.DataFrame, spec: ChartSpec) -> None:
    """Rescaling onto a second axis lives in the renderer, not the values."""
    frame = frame.assign(other=[1.0, 2.0, 3.0, 4.0, 5.0])
    result = scores_for(frame, spec, DualAxis(columns=("other",)))
    assert result.slope_ratio == pytest.approx(1.0)
    assert result.data_fidelity == 1.0
