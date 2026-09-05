from __future__ import annotations

import pytest

from core.ingest import ColumnType, suggest_spec
from core.samples import list_samples, load_sample
from core.spec import ChartFamily


@pytest.mark.parametrize("sample", list_samples(), ids=lambda s: s.key)
def test_every_sample_loads_and_yields_a_spec(sample) -> None:
    dataset = load_sample(sample.key)
    assert not dataset.frame.empty
    assert dataset.source_name == sample.filename
    assert suggest_spec(dataset).is_baseline


def test_unknown_sample_raises() -> None:
    with pytest.raises(KeyError):
        load_sample("nope")


def test_mrr_sample_is_a_monthly_series() -> None:
    dataset = load_sample("saas_mrr")
    assert len(dataset.frame) == 36
    assert dataset.types["month"] is ColumnType.DATETIME
    assert dataset.types["mrr_usd"] is ColumnType.NUMERIC
    assert dataset.issues == ()

    spec = suggest_spec(dataset)
    assert spec.family is ChartFamily.LINE
    assert spec.x == "month"


def test_mrr_sample_contains_the_dip_the_demos_rely_on() -> None:
    mrr = load_sample("saas_mrr").frame["mrr_usd"]
    assert mrr.iloc[-1] > mrr.iloc[0]  # the honest story is growth
    assert mrr.diff().min() < 0  # but there is a fall to cherry-pick


def test_ticket_sample_is_noisy_enough_to_smooth() -> None:
    tickets = load_sample("support_tickets").frame["tickets"]
    assert len(tickets) == 730
    assert tickets.std() > 20


def test_regional_sample_has_the_long_tail_the_demos_rely_on() -> None:
    frame = load_sample("regional_sales").frame
    by_region = frame.groupby("region")["revenue_usd"].sum().sort_values()
    assert by_region.iloc[-1] > by_region.iloc[0] * 10
    assert by_region.mean() > by_region.median()  # mean is dragged by the leaders


def test_messy_sample_trips_every_validation_rule() -> None:
    dataset = load_sample("messy_inventory")
    codes = {issue.code for issue in dataset.issues}
    assert {"duplicate_header", "all_null", "mixed_types", "sparse"} <= codes
    assert len(set(dataset.frame.columns)) == len(dataset.frame.columns)
