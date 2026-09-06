from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.prepare import SINGLE_SERIES, PrepareError, prepare
from core.spec import Aggregation, ChartFamily, ChartSpec, ValueMode


@pytest.fixture
def monthly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]),
            "revenue": [100.0, 200.0, 150.0, 250.0],
        }
    )


def line_spec(**changes: object) -> ChartSpec:
    spec = ChartSpec(family=ChartFamily.LINE, x="month", y="revenue")
    return spec.evolve(**changes) if changes else spec


def test_baseline_returns_one_point_per_x(monthly: pd.DataFrame) -> None:
    prepared = prepare(monthly, line_spec())
    assert prepared.values.tolist() == [100.0, 200.0, 150.0, 250.0]
    assert prepared.series_names == (SINGLE_SERIES,)
    assert prepared.dropped_rows == 0


def test_baseline_is_zero_based(monthly: pd.DataFrame) -> None:
    assert prepare(monthly, line_spec()).y_domain == (0.0, 250.0)


def test_negative_values_extend_the_domain_below_zero(monthly: pd.DataFrame) -> None:
    frame = monthly.assign(revenue=[-50.0, 10.0, 20.0, 30.0])
    assert prepare(frame, line_spec()).y_domain[0] == -50.0


def test_explicit_bounds_win(monthly: pd.DataFrame) -> None:
    assert prepare(monthly, line_spec(y_min=90.0, y_max=300.0)).y_domain == (90.0, 300.0)


def test_a_flat_series_still_gets_a_usable_domain() -> None:
    frame = pd.DataFrame({"month": ["a", "b"], "revenue": [0.0, 0.0]})
    low, high = prepare(frame, ChartSpec(family=ChartFamily.BAR, x="month", y="revenue")).y_domain
    assert high > low


def test_duplicate_x_values_are_aggregated(monthly: pd.DataFrame) -> None:
    doubled = pd.concat([monthly, monthly], ignore_index=True)
    assert prepare(doubled, line_spec()).values.tolist() == [200.0, 400.0, 300.0, 500.0]


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        (Aggregation.SUM, 30.0),
        (Aggregation.MEAN, 10.0),
        (Aggregation.MEDIAN, 5.0),
        (Aggregation.COUNT, 3.0),
    ],
)
def test_each_aggregation(aggregation: Aggregation, expected: float) -> None:
    frame = pd.DataFrame({"g": ["a", "a", "a"], "v": [1.0, 5.0, 24.0]})
    spec = ChartSpec(family=ChartFamily.BAR, x="g", y="v", aggregation=aggregation)
    assert prepare(frame, spec).values.tolist() == [expected]


def test_count_needs_no_measure_column() -> None:
    frame = pd.DataFrame({"g": ["a", "a", "b"]})
    spec = ChartSpec(family=ChartFamily.BAR, x="g", aggregation=Aggregation.COUNT)
    assert prepare(frame, spec).values.tolist() == [2.0, 1.0]


def test_series_split_produces_one_line_each() -> None:
    frame = pd.DataFrame(
        {"m": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0], "r": ["n", "s", "n", "s"]}
    )
    prepared = prepare(frame, ChartSpec(family=ChartFamily.LINE, x="m", y="v", series="r"))
    assert prepared.series_names == ("n", "s")
    assert prepared.series("n").tolist() == [1.0, 3.0]


def test_datetime_window_is_inclusive(monthly: pd.DataFrame) -> None:
    prepared = prepare(monthly, line_spec(x_range=("2024-02-01", "2024-03-01")))
    assert prepared.values.tolist() == [200.0, 150.0]
    assert prepared.dropped_rows == 2


def test_an_open_ended_window_keeps_the_far_end(monthly: pd.DataFrame) -> None:
    assert prepare(monthly, line_spec(x_range=("2024-03-01", None))).values.tolist() == [
        150.0,
        250.0,
    ]


def test_numeric_window() -> None:
    frame = pd.DataFrame({"i": [1, 2, 3, 4], "v": [10.0, 20.0, 30.0, 40.0]})
    spec = ChartSpec(family=ChartFamily.LINE, x="i", y="v", x_range=(2, 3))
    assert prepare(frame, spec).values.tolist() == [20.0, 30.0]


def test_categorical_window_slices_by_appearance_order() -> None:
    frame = pd.DataFrame({"q": ["q1", "q2", "q3", "q4"], "v": [1.0, 2.0, 3.0, 4.0]})
    spec = ChartSpec(family=ChartFamily.BAR, x="q", y="v", x_range=("q2", "q3"))
    assert prepare(frame, spec).values.tolist() == [2.0, 3.0]


def test_an_unknown_categorical_bound_is_rejected() -> None:
    frame = pd.DataFrame({"q": ["q1", "q2"], "v": [1.0, 2.0]})
    spec = ChartSpec(family=ChartFamily.BAR, x="q", y="v", x_range=("q9", None))
    with pytest.raises(PrepareError, match="not a value of"):
        prepare(frame, spec)


def test_an_empty_window_is_rejected(monthly: pd.DataFrame) -> None:
    with pytest.raises(PrepareError, match="no rows"):
        prepare(monthly, line_spec(x_range=("2030-01-01", "2030-02-01")))


def test_smoothing_averages_a_trailing_window(monthly: pd.DataFrame) -> None:
    values = prepare(monthly, line_spec(smoothing_window=2)).values.tolist()
    assert values == [100.0, 150.0, 175.0, 200.0]


def test_smoothing_is_per_series() -> None:
    frame = pd.DataFrame(
        {"m": ["a", "a", "b", "b"], "v": [10.0, 100.0, 20.0, 200.0], "r": ["n", "s", "n", "s"]}
    )
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v", series="r", smoothing_window=2)
    prepared = prepare(frame, spec)
    assert prepared.series("n").tolist() == [10.0, 15.0]
    assert prepared.series("s").tolist() == [100.0, 150.0]


def test_cumulative_only_ever_rises(monthly: pd.DataFrame) -> None:
    values = prepare(monthly, line_spec(cumulative=True)).values
    assert values.tolist() == [100.0, 300.0, 450.0, 700.0]
    assert (values.diff().dropna() >= 0).all()


def test_percent_change_restates_the_first_point_as_zero(monthly: pd.DataFrame) -> None:
    values = prepare(monthly, line_spec(value_mode=ValueMode.PERCENT_CHANGE)).values
    assert values.iloc[0] == 0.0
    assert values.iloc[1] == pytest.approx(100.0)
    assert values.iloc[2] == pytest.approx(-25.0)


def test_indexing_rebases_on_the_anchor(monthly: pd.DataFrame) -> None:
    spec = line_spec(value_mode=ValueMode.INDEXED, index_anchor="2024-02-01")
    values = prepare(monthly, spec).values
    assert values.iloc[1] == pytest.approx(100.0)
    assert values.iloc[3] == pytest.approx(125.0)


def test_indexing_on_a_missing_anchor_is_rejected(monthly: pd.DataFrame) -> None:
    spec = line_spec(value_mode=ValueMode.INDEXED, index_anchor="2030-01-01")
    with pytest.raises(PrepareError, match="anchor"):
        prepare(monthly, spec)


def test_indexing_on_a_zero_value_is_rejected(monthly: pd.DataFrame) -> None:
    frame = monthly.assign(revenue=[0.0, 1.0, 2.0, 3.0])
    spec = line_spec(value_mode=ValueMode.INDEXED, index_anchor="2024-01-01")
    with pytest.raises(PrepareError, match="zero value"):
        prepare(frame, spec)


def test_steps_compose_in_the_documented_order(monthly: pd.DataFrame) -> None:
    # window -> aggregate -> smooth -> accumulate: the last three points,
    # smoothed pairwise, then accumulated.
    spec = line_spec(x_range=("2024-02-01", None), smoothing_window=2, cumulative=True)
    assert prepare(monthly, spec).values.tolist() == [200.0, 375.0, 575.0]


def test_infinities_do_not_poison_the_domain(monthly: pd.DataFrame) -> None:
    frame = monthly.assign(revenue=[100.0, np.inf, 150.0, 250.0])
    assert prepare(frame, line_spec()).y_domain == (0.0, 250.0)


def test_a_missing_column_is_rejected(monthly: pd.DataFrame) -> None:
    with pytest.raises(PrepareError, match="columns not in the data"):
        prepare(monthly, line_spec(x="nope"))


def test_a_non_numeric_measure_is_rejected(monthly: pd.DataFrame) -> None:
    frame = monthly.assign(label=["a", "b", "c", "d"])
    with pytest.raises(PrepareError, match="not numeric"):
        prepare(frame, line_spec(y="label"))


def test_series_lookup_and_emptiness(monthly: pd.DataFrame) -> None:
    prepared = prepare(monthly, line_spec())
    assert not prepared.is_empty
    assert prepared.series(SINGLE_SERIES).tolist() == [100.0, 200.0, 150.0, 250.0]
