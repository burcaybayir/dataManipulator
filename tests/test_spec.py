from __future__ import annotations

import pytest

from core.spec import Aggregation, ChartFamily, ChartSpec, ValueMode


def base_spec(**changes: object) -> ChartSpec:
    spec = ChartSpec(family=ChartFamily.LINE, x="month", y="revenue")
    return spec.evolve(**changes) if changes else spec


def test_defaults_are_baseline() -> None:
    assert base_spec().is_baseline


@pytest.mark.parametrize(
    "changes",
    [
        {"y_min": 900.0},
        {"invert_y": True},
        {"x_range": ("2024-01-01", "2024-03-01")},
        {"smoothing_window": 3},
        {"cumulative": True},
        {"aspect_ratio": (4.0, 1.0)},
        {"secondary_y": ("revenue",)},
    ],
)
def test_any_distortion_knob_clears_baseline(changes: dict[str, object]) -> None:
    assert not base_spec(**changes).is_baseline


def test_evolve_leaves_the_original_untouched() -> None:
    original = base_spec()
    distorted = original.evolve(y_min=900.0)
    assert original.y_min is None
    assert distorted.y_min == 900.0
    assert original.is_baseline


def test_count_aggregation_may_omit_y() -> None:
    spec = ChartSpec(family=ChartFamily.BAR, x="region", aggregation=Aggregation.COUNT)
    assert spec.y is None
    assert spec.columns == ("region",)


def test_missing_y_without_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="COUNT"):
        ChartSpec(family=ChartFamily.BAR, x="region")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"smoothing_window": 1}, "at least 2"),
        ({"y_min": 10.0, "y_max": 5.0}, "below y_max"),
        ({"aspect_ratio": (0.0, 9.0)}, "positive"),
        ({"value_mode": ValueMode.INDEXED}, "index_anchor"),
    ],
)
def test_invalid_parameters_are_rejected(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        base_spec(**changes)


def test_columns_deduplicates_and_preserves_order() -> None:
    spec = ChartSpec(family=ChartFamily.BAR, x="region", y="revenue", series="region")
    assert spec.columns == ("region", "revenue")


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        (ChartFamily.BAR, True),
        (ChartFamily.AREA, True),
        (ChartFamily.PIE, True),
        (ChartFamily.LINE, False),
        (ChartFamily.SCATTER, False),
    ],
)
def test_length_encoded_families_need_a_zero_baseline(family: ChartFamily, expected: bool) -> None:
    assert base_spec(family=family).requires_zero_baseline is expected


def test_round_trips_through_a_manifest() -> None:
    spec = base_spec(
        y_min=900.0,
        x_range=("2024-01-01", "2024-06-01"),
        smoothing_window=3,
        secondary_y=("revenue",),
        notes=("truncated_axis",),
    )
    assert ChartSpec.from_dict(spec.to_dict()) == spec


def test_manifest_is_json_safe() -> None:
    import json

    payload = json.dumps(base_spec(smoothing_window=4).to_dict())
    assert ChartSpec.from_dict(json.loads(payload)).smoothing_window == 4
