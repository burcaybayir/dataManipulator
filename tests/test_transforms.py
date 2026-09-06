"""Tests for the transform library.

Every transform is checked for three things: that it changes what it claims to
change, that it leaves its inputs untouched, and that it round-trips through a
manifest entry.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.prepare import prepare
from core.spec import Aggregation, ChartFamily, ChartSpec, ValueMode
from core.transforms import (
    AggregationSwap,
    AspectRatioChange,
    BinRegrouping,
    CherryPickedWindow,
    Cumulative,
    DualAxis,
    InvertedAxis,
    OutlierDrop,
    RatioVsAbsolute,
    RebaseIndex,
    Smoothing,
    Transform,
    TransformError,
    TruncatedAxis,
    available,
    build,
    from_dict,
    get,
    register,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
            ),
            "revenue": [100.0, 120.0, 90.0, 130.0, 140.0],
            "headcount": [10.0, 11.0, 12.0, 13.0, 14.0],
            "region": ["north", "north", "south", "south", "east"],
        }
    )


@pytest.fixture
def spec() -> ChartSpec:
    return ChartSpec(family=ChartFamily.LINE, x="month", y="revenue")


ALL_TRANSFORMS: list[Transform] = [
    TruncatedAxis(),
    InvertedAxis(),
    CherryPickedWindow(start="2024-02-01", end="2024-04-01"),
    AggregationSwap(target=Aggregation.MEDIAN),
    RatioVsAbsolute(),
    RebaseIndex(),
    BinRegrouping(column="region", keep_top=1),
    Smoothing(window=2),
    Cumulative(),
    DualAxis(columns=("headcount",)),
    AspectRatioChange(),
    OutlierDrop(z_threshold=1.0),
]


# --- contract shared by every transform -------------------------------------


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=lambda t: t.name)
def test_every_transform_documents_itself(transform: Transform) -> None:
    assert transform.label
    assert transform.explanation
    assert transform.why_misleading
    assert transform.legitimate_when
    assert transform.name in transform.to_dict()["transform"]


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=lambda t: t.name)
def test_every_transform_leaves_its_inputs_untouched(
    transform: Transform, frame: pd.DataFrame, spec: ChartSpec
) -> None:
    before = frame.copy()
    transform.apply(frame, spec)
    pd.testing.assert_frame_equal(frame, before)
    assert spec.is_baseline


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=lambda t: t.name)
def test_every_transform_records_a_note(
    transform: Transform, frame: pd.DataFrame, spec: ChartSpec
) -> None:
    _, out = transform.apply(frame, spec)
    assert out.notes == (transform.note(),)


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=lambda t: t.name)
def test_every_transform_round_trips_through_a_manifest(transform: Transform) -> None:
    import json

    payload = json.loads(json.dumps(transform.to_dict()))
    assert from_dict(payload) == transform


@pytest.mark.parametrize("transform", ALL_TRANSFORMS, ids=lambda t: t.name)
def test_every_transform_changes_something(
    transform: Transform, frame: pd.DataFrame, spec: ChartSpec
) -> None:
    out_frame, out_spec = transform.apply(frame, spec)
    changed_spec = out_spec.evolve(notes=()) != spec
    changed_data = not out_frame.equals(frame)
    assert changed_spec or changed_data


def test_every_registered_transform_is_covered_by_these_tests() -> None:
    assert {type(t) for t in ALL_TRANSFORMS} == set(available())


# --- individual behaviours ---------------------------------------------------


def test_truncated_axis_lifts_the_floor_off_zero(frame: pd.DataFrame, spec: ChartSpec) -> None:
    assert prepare(frame, spec).y_domain[0] == 0.0
    _, out = TruncatedAxis().apply(frame, spec)
    low, _ = prepare(frame, out).y_domain
    assert 0 < low < 90.0


def test_truncated_axis_accepts_an_explicit_floor(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = TruncatedAxis(floor=85.0).apply(frame, spec)
    assert out.y_min == 85.0


def test_a_smaller_headroom_truncates_harder(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, tight = TruncatedAxis(headroom=0.01).apply(frame, spec)
    _, loose = TruncatedAxis(headroom=0.5).apply(frame, spec)
    assert tight.y_min is not None and loose.y_min is not None
    assert tight.y_min > loose.y_min


def test_truncated_axis_handles_a_flat_series(spec: ChartSpec) -> None:
    flat = pd.DataFrame({"month": ["a", "b"], "revenue": [50.0, 50.0]})
    _, out = TruncatedAxis().apply(flat, spec.evolve(x="month"))
    assert out.y_min is not None and out.y_min < 50.0


def test_truncated_axis_rejects_a_negative_headroom() -> None:
    with pytest.raises(TransformError, match="headroom"):
        TruncatedAxis(headroom=-1.0)


def test_truncated_axis_rejects_a_floor_above_the_ceiling(
    frame: pd.DataFrame, spec: ChartSpec
) -> None:
    with pytest.raises(TransformError, match="axis maximum"):
        TruncatedAxis(floor=500.0).apply(frame, spec.evolve(y_max=200.0))


def test_truncated_axis_needs_values_to_measure(spec: ChartSpec) -> None:
    blank = pd.DataFrame({"month": ["a"], "revenue": [float("nan")]})
    with pytest.raises(TransformError, match="no values"):
        TruncatedAxis().apply(blank, spec.evolve(x="month", aggregation=Aggregation.MEAN))


def test_a_transform_reports_preparation_failures_as_its_own(
    frame: pd.DataFrame, spec: ChartSpec
) -> None:
    empty = frame.iloc[:0]
    with pytest.raises(TransformError, match="no rows"):
        TruncatedAxis().apply(empty, spec)


def test_inverted_axis_toggles(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, once = InvertedAxis().apply(frame, spec)
    assert once.invert_y
    _, twice = InvertedAxis().apply(frame, once)
    assert not twice.invert_y


def test_cherry_picked_window_narrows_the_data(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = CherryPickedWindow(start="2024-02-01", end="2024-03-01").apply(frame, spec)
    assert prepare(frame, out).values.tolist() == [120.0, 90.0]


def test_a_window_needs_at_least_one_bound() -> None:
    with pytest.raises(TransformError, match="start, an end"):
        CherryPickedWindow()


def test_favouring_down_finds_the_falling_stretch(frame: pd.DataFrame, spec: ChartSpec) -> None:
    window = CherryPickedWindow.favouring(frame, spec, direction="down", length=2)
    _, out = window.apply(frame, spec)
    values = prepare(frame, out).values.tolist()
    assert values == [120.0, 90.0]


def test_favouring_up_finds_the_rising_stretch(frame: pd.DataFrame, spec: ChartSpec) -> None:
    window = CherryPickedWindow.favouring(frame, spec, direction="up", length=2)
    _, out = window.apply(frame, spec)
    values = prepare(frame, out).values.tolist()
    assert values[-1] > values[0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"direction": "sideways"}, "direction"),
        ({"direction": "up", "length": 1}, "two points"),
        ({"direction": "up", "length": 99}, "points available"),
    ],
)
def test_favouring_rejects_bad_arguments(
    frame: pd.DataFrame, spec: ChartSpec, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(TransformError, match=message):
        CherryPickedWindow.favouring(frame, spec, **kwargs)  # type: ignore[arg-type]


def test_favouring_labels_a_categorical_axis() -> None:
    frame = pd.DataFrame({"q": ["q1", "q2", "q3"], "v": [3.0, 2.0, 1.0]})
    spec = ChartSpec(family=ChartFamily.BAR, x="q", y="v")
    window = CherryPickedWindow.favouring(frame, spec, direction="down", length=2)
    assert (window.start, window.end) == ("q1", "q2")


def test_aggregation_swap_changes_the_level() -> None:
    frame = pd.DataFrame({"g": ["a", "a", "a"], "v": [1.0, 2.0, 60.0]})
    spec = ChartSpec(family=ChartFamily.BAR, x="g", y="v", aggregation=Aggregation.MEAN)
    _, out = AggregationSwap(target=Aggregation.MEDIAN).apply(frame, spec)
    assert prepare(frame, spec).values.tolist() == [21.0]
    assert prepare(frame, out).values.tolist() == [2.0]


def test_aggregation_swap_accepts_a_string_parameter() -> None:
    assert AggregationSwap(target="median").target is Aggregation.MEDIAN  # type: ignore[arg-type]


def test_aggregation_swap_needs_a_measure(frame: pd.DataFrame) -> None:
    counting = ChartSpec(family=ChartFamily.BAR, x="region", aggregation=Aggregation.COUNT)
    with pytest.raises(TransformError, match="needs a measure"):
        AggregationSwap(target=Aggregation.SUM).apply(frame, counting)


def test_ratio_flatters_a_small_base() -> None:
    frame = pd.DataFrame({"m": ["a", "b"], "v": [3.0, 6.0], "r": ["small", "small"]})
    spec = ChartSpec(family=ChartFamily.LINE, x="m", y="v")
    _, out = RatioVsAbsolute().apply(frame, spec)
    assert prepare(frame, out).values.tolist() == [0.0, 100.0]


def test_ratio_can_switch_back_to_absolute(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = RatioVsAbsolute(mode=ValueMode.ABSOLUTE).apply(frame, spec)
    assert out.value_mode is ValueMode.ABSOLUTE


def test_ratio_refuses_to_impersonate_rebasing() -> None:
    with pytest.raises(TransformError, match="rebase_index"):
        RatioVsAbsolute(mode=ValueMode.INDEXED)


def test_rebase_defaults_to_the_first_point(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = RebaseIndex().apply(frame, spec)
    assert out.value_mode is ValueMode.INDEXED
    assert prepare(frame, out).values.iloc[0] == pytest.approx(100.0)


def test_rebase_on_a_chosen_anchor_moves_the_baseline(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = RebaseIndex(anchor="2024-03-01").apply(frame, spec)
    values = prepare(frame, out).values
    assert values.iloc[2] == pytest.approx(100.0)
    assert values.iloc[0] > 100.0


def test_rebase_needs_a_point_to_anchor_on(spec: ChartSpec) -> None:
    empty = pd.DataFrame({"month": pd.to_datetime([]), "revenue": []})
    with pytest.raises(TransformError, match="no rows"):
        RebaseIndex().apply(empty, spec)


def test_bin_regrouping_folds_the_tail(frame: pd.DataFrame) -> None:
    # north and south total 220 each; east trails on 140.
    spec = ChartSpec(family=ChartFamily.BAR, x="region", y="revenue")
    out_frame, _ = BinRegrouping(keep_top=2).apply(frame, spec)
    assert set(out_frame["region"]) == {"north", "south", "Other"}
    assert out_frame.loc[out_frame["region"] == "Other", "revenue"].tolist() == [140.0]


def test_bin_regrouping_applies_an_explicit_mapping(frame: pd.DataFrame) -> None:
    spec = ChartSpec(family=ChartFamily.BAR, x="region", y="revenue")
    mapping = (("north", "EMEA"), ("south", "EMEA"))
    out_frame, _ = BinRegrouping(mapping=mapping).apply(frame, spec)
    assert set(out_frame["region"]) == {"EMEA", "east"}


def test_bin_regrouping_ranks_by_count_when_counting(frame: pd.DataFrame) -> None:
    spec = ChartSpec(family=ChartFamily.BAR, x="region", aggregation=Aggregation.COUNT)
    out_frame, _ = BinRegrouping(keep_top=1).apply(frame, spec)
    assert "north" in set(out_frame["region"])


def test_bin_regrouping_defaults_to_the_series_column(frame: pd.DataFrame) -> None:
    spec = ChartSpec(family=ChartFamily.LINE, x="month", y="revenue", series="region")
    out_frame, _ = BinRegrouping(keep_top=1).apply(frame, spec)
    assert "Other" in set(out_frame["region"])


def test_bin_regrouping_serializes_its_mapping_as_json() -> None:
    transform = BinRegrouping(mapping=(("a", "b"),))
    assert transform.to_dict()["params"]["mapping"] == [["a", "b"]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mapping": (("a",),)}, "pairs"),
        ({"keep_top": 0}, "keep_top"),
        ({"keep_top": None}, "keep_top"),
    ],
)
def test_bin_regrouping_rejects_bad_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(TransformError, match=message):
        BinRegrouping(**kwargs)  # type: ignore[arg-type]


def test_bin_regrouping_needs_a_categorical_column(frame: pd.DataFrame, spec: ChartSpec) -> None:
    with pytest.raises(TransformError, match="not categorical"):
        BinRegrouping(column="revenue").apply(frame, spec)


def test_bin_regrouping_needs_a_column_that_exists(frame: pd.DataFrame, spec: ChartSpec) -> None:
    with pytest.raises(TransformError, match="not in the data"):
        BinRegrouping(column="nope").apply(frame, spec)


def test_smoothing_reduces_volatility(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = Smoothing(window=3).apply(frame, spec)
    assert prepare(frame, out).values.std() < prepare(frame, spec).values.std()


def test_smoothing_needs_a_real_window() -> None:
    with pytest.raises(TransformError, match="two points"):
        Smoothing(window=1)


def test_cumulative_never_falls_for_positive_data(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = Cumulative().apply(frame, spec)
    values = prepare(frame, out).values
    assert (values.diff().dropna() >= 0).all()


def test_dual_axis_picks_a_spare_measure(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = DualAxis().apply(frame, spec)
    assert out.secondary_y == ("headcount",)


def test_dual_axis_rejects_unknown_columns(frame: pd.DataFrame, spec: ChartSpec) -> None:
    with pytest.raises(TransformError, match="not in the data"):
        DualAxis(columns=("nope",)).apply(frame, spec)


def test_dual_axis_needs_a_spare_measure(spec: ChartSpec) -> None:
    frame = pd.DataFrame({"month": ["a", "b"], "revenue": [1.0, 2.0]})
    with pytest.raises(TransformError, match="no spare numeric"):
        DualAxis().apply(frame, spec.evolve(x="month"))


def test_aspect_ratio_reshapes_the_frame(frame: pd.DataFrame, spec: ChartSpec) -> None:
    _, out = AspectRatioChange(width=4.0, height=1.0).apply(frame, spec)
    assert out.aspect_ratio == (4.0, 1.0)


def test_a_tall_frame_steepens_and_a_wide_one_flattens() -> None:
    assert AspectRatioChange(width=6.0, height=9.0).steepening
    assert not AspectRatioChange(width=32.0, height=9.0).steepening


def test_aspect_ratio_rejects_impossible_shapes() -> None:
    with pytest.raises(TransformError, match="positive"):
        AspectRatioChange(width=0.0, height=1.0)


def test_outlier_drop_removes_the_spike(spec: ChartSpec) -> None:
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime([f"2024-0{i}-01" for i in range(1, 7)]),
            "revenue": [10.0, 11.0, 12.0, 11.0, 10.0, 900.0],
        }
    )
    out_frame, _ = OutlierDrop(z_threshold=1.5).apply(frame, spec)
    assert 900.0 not in out_frame["revenue"].tolist()
    assert len(out_frame) == 5


def test_outlier_drop_is_a_no_op_on_a_flat_series(spec: ChartSpec) -> None:
    frame = pd.DataFrame({"month": ["a", "b"], "revenue": [5.0, 5.0]})
    out_frame, _ = OutlierDrop().apply(frame, spec.evolve(x="month"))
    assert len(out_frame) == 2


def test_outlier_drop_refuses_to_empty_the_data(spec: ChartSpec) -> None:
    frame = pd.DataFrame({"month": ["a", "b"], "revenue": [0.0, 100.0]})
    with pytest.raises(TransformError, match="every row"):
        OutlierDrop(z_threshold=0.5).apply(frame, spec.evolve(x="month"))


def test_outlier_drop_needs_a_numeric_measure(frame: pd.DataFrame, spec: ChartSpec) -> None:
    with pytest.raises(TransformError, match="not numeric"):
        OutlierDrop(column="region").apply(frame, spec)


def test_outlier_drop_needs_a_column_that_exists(frame: pd.DataFrame, spec: ChartSpec) -> None:
    with pytest.raises(TransformError, match="not in the data"):
        OutlierDrop(column="nope").apply(frame, spec)


def test_outlier_drop_needs_a_measure_at_all(frame: pd.DataFrame) -> None:
    counting = ChartSpec(family=ChartFamily.BAR, x="region", aggregation=Aggregation.COUNT)
    with pytest.raises(TransformError, match="no measure column"):
        OutlierDrop().apply(frame, counting)


def test_outlier_drop_rejects_a_non_positive_threshold() -> None:
    with pytest.raises(TransformError, match="positive"):
        OutlierDrop(z_threshold=0.0)


# --- registry ----------------------------------------------------------------


def test_the_registry_holds_the_twelve_documented_techniques() -> None:
    assert len(available()) == 12
    assert get("truncated_axis") is TruncatedAxis


def test_build_constructs_from_name_and_params() -> None:
    assert build("smoothing", window=4) == Smoothing(window=4)


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(TransformError, match="unknown transform"):
        get("nope")


def test_bad_parameters_are_rejected() -> None:
    with pytest.raises(TransformError, match="bad parameters"):
        build("smoothing", nonsense=1)


def test_a_manifest_entry_without_a_name_is_rejected() -> None:
    with pytest.raises(TransformError, match="no 'transform' key"):
        from_dict({"params": {}})


def test_registering_a_nameless_transform_is_rejected() -> None:
    class Nameless(Transform):
        def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
            return frame, spec

    with pytest.raises(TransformError, match="must define a name"):
        register(Nameless)


def test_registering_a_duplicate_name_is_rejected() -> None:
    class Clash(Transform):
        name = "smoothing"

        def apply(self, frame: pd.DataFrame, spec: ChartSpec) -> tuple[pd.DataFrame, ChartSpec]:
            return frame, spec

    with pytest.raises(TransformError, match="already registered"):
        register(Clash)


def test_describe_names_the_technique_and_its_parameters() -> None:
    assert Smoothing(window=3).describe() == "Smoothed series (window=3)"
    assert Cumulative().describe() == "Cumulative total"
    assert CherryPickedWindow(start="a").describe() == "Cherry-picked window (start='a')"


def test_timestamp_parameters_serialize_as_iso_strings() -> None:
    transform = CherryPickedWindow(start=pd.Timestamp("2024-01-01"))
    assert transform.to_dict()["params"]["start"] == "2024-01-01T00:00:00"


def test_params_of_a_transform_without_fields() -> None:
    assert Cumulative().params == {}


def test_modifies_data_flags_the_row_changing_transforms() -> None:
    assert BinRegrouping().modifies_data
    assert OutlierDrop().modifies_data
    assert not Smoothing().modifies_data
