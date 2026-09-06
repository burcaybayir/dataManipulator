from __future__ import annotations

import pandas as pd
import pytest

from core.prepare import prepare
from core.spec import ChartFamily, ChartSpec
from core.transforms import (
    MAX_DEPTH,
    BinRegrouping,
    Cumulative,
    InvertedAxis,
    Smoothing,
    TransformError,
    TransformStack,
    TruncatedAxis,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": pd.to_datetime([f"2024-0{i}-01" for i in range(1, 6)]),
            "revenue": [100.0, 120.0, 90.0, 130.0, 140.0],
            "region": ["north", "north", "south", "south", "east"],
        }
    )


@pytest.fixture
def spec() -> ChartSpec:
    return ChartSpec(family=ChartFamily.LINE, x="month", y="revenue")


def test_an_empty_stack_is_the_baseline(frame: pd.DataFrame, spec: ChartSpec) -> None:
    stack = TransformStack()
    out_frame, out_spec = stack.apply(frame, spec)
    assert stack.is_empty
    assert not stack
    assert out_spec == spec
    pd.testing.assert_frame_equal(out_frame, frame)


def test_push_returns_a_new_stack(spec: ChartSpec) -> None:
    first = TransformStack()
    second = first.push(Smoothing(window=2))
    assert len(first) == 0
    assert len(second) == 1
    assert second.names == ("smoothing",)


def test_push_accepts_several_transforms_at_once() -> None:
    stack = TransformStack().push(Smoothing(window=2), Cumulative())
    assert stack.names == ("smoothing", "cumulative")
    assert list(stack) == [Smoothing(window=2), Cumulative()]


def test_transforms_apply_in_order(frame: pd.DataFrame, spec: ChartSpec) -> None:
    smooth_then_sum = TransformStack().push(Smoothing(window=2), Cumulative())
    sum_then_smooth = TransformStack().push(Cumulative(), Smoothing(window=2))
    # Both set the same two spec fields, so prepare's fixed order makes the
    # plotted values identical; the recorded provenance is what differs.
    assert smooth_then_sum.describe() != sum_then_smooth.describe()
    assert smooth_then_sum.apply(frame, spec)[1].notes[0].startswith("Smoothed")


def test_a_stack_composes_its_effects(frame: pd.DataFrame, spec: ChartSpec) -> None:
    stack = TransformStack().push(TruncatedAxis(), InvertedAxis(), Smoothing(window=2))
    _, out = stack.apply(frame, spec)
    assert out.y_min is not None
    assert out.invert_y
    assert out.smoothing_window == 2
    assert len(out.notes) == 3
    assert not out.is_baseline


def test_a_stack_can_change_both_data_and_spec(frame: pd.DataFrame, spec: ChartSpec) -> None:
    stack = TransformStack().push(BinRegrouping(column="region", keep_top=1), Smoothing(window=2))
    out_frame, out_spec = stack.apply(frame, spec)
    assert stack.modifies_data
    assert "Other" in set(out_frame["region"])
    assert out_spec.smoothing_window == 2


def test_a_spec_only_stack_reports_that_it_leaves_data_alone() -> None:
    assert not TransformStack().push(Smoothing(window=2)).modifies_data


def test_a_failing_step_names_its_position(frame: pd.DataFrame, spec: ChartSpec) -> None:
    stack = TransformStack().push(Smoothing(window=2), BinRegrouping(column="revenue"))
    with pytest.raises(TransformError, match=r"step 2 \(bin_regrouping\)"):
        stack.apply(frame, spec)


def test_the_depth_limit_is_enforced() -> None:
    too_many = [Smoothing(window=w) for w in range(2, 2 + MAX_DEPTH + 1)]
    with pytest.raises(TransformError, match="depth limit"):
        TransformStack(tuple(too_many))


def test_a_stack_at_the_limit_is_allowed() -> None:
    at_limit = [Smoothing(window=w) for w in range(2, 2 + MAX_DEPTH)]
    assert len(TransformStack(tuple(at_limit))) == MAX_DEPTH


def test_describe_lists_every_step() -> None:
    stack = TransformStack().push(Smoothing(window=3), Cumulative())
    assert stack.describe() == ("Smoothed series (window=3)", "Cumulative total")


def test_a_stack_round_trips_through_a_manifest() -> None:
    import json

    stack = TransformStack().push(TruncatedAxis(floor=10.0), BinRegrouping(keep_top=2))
    assert TransformStack.from_dict(json.loads(json.dumps(stack.to_dict()))) == stack


def test_the_same_stack_applied_twice_gives_the_same_values(
    frame: pd.DataFrame, spec: ChartSpec
) -> None:
    stack = TransformStack().push(BinRegrouping(column="region", keep_top=1), Smoothing(window=2))
    first = prepare(*stack.apply(frame, spec))
    second = prepare(*stack.apply(frame, spec))
    pd.testing.assert_frame_equal(first.frame, second.frame)
