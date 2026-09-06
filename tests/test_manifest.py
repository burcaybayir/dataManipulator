from __future__ import annotations

import pandas as pd
import pytest

from core.ingest import load, suggest_spec
from core.manifest import FORMAT_VERSION, Manifest, ManifestError
from core.samples import load_sample
from core.spec import ChartFamily, ChartSpec
from core.transforms import Smoothing, TransformStack, TruncatedAxis


@pytest.fixture
def stack() -> TransformStack:
    return TransformStack().push(TruncatedAxis(headroom=0.02), Smoothing(window=3))


def test_a_manifest_records_the_data_it_was_built_from(stack: TransformStack) -> None:
    dataset = load_sample("saas_mrr")
    manifest = Manifest.for_dataset(dataset, suggest_spec(dataset), stack)
    assert manifest.source_hash == dataset.source_hash
    assert manifest.source_name == "saas_mrr.csv"


def test_a_manifest_round_trips_through_json(stack: TransformStack) -> None:
    dataset = load_sample("saas_mrr")
    manifest = Manifest.for_dataset(dataset, suggest_spec(dataset), stack)
    assert Manifest.from_json(manifest.to_json()) == manifest


def test_replaying_a_manifest_reproduces_the_chart(stack: TransformStack) -> None:
    dataset = load_sample("saas_mrr")
    manifest = Manifest.for_dataset(dataset, suggest_spec(dataset), stack)
    replayed = Manifest.from_json(manifest.to_json()).render_dataset(dataset)
    pd.testing.assert_frame_equal(replayed.frame, manifest.render_dataset(dataset).frame)
    assert replayed.y_domain == manifest.render_dataset(dataset).y_domain


def test_replaying_against_different_data_is_refused(stack: TransformStack) -> None:
    dataset = load_sample("saas_mrr")
    manifest = Manifest.for_dataset(dataset, suggest_spec(dataset), stack)
    other = load(dataset.frame.head(10).to_csv(index=False).encode(), name="trimmed.csv")
    with pytest.raises(ManifestError, match="hashes to"):
        manifest.render_dataset(other)


def test_the_hash_check_can_be_waived(stack: TransformStack) -> None:
    dataset = load_sample("saas_mrr")
    manifest = Manifest.for_dataset(dataset, suggest_spec(dataset), stack)
    other = load(dataset.frame.head(10).to_csv(index=False).encode(), name="trimmed.csv")
    assert len(manifest.render_dataset(other, check_hash=False).frame) == 10


def test_a_manifest_without_a_stack_renders_the_baseline() -> None:
    dataset = load_sample("saas_mrr")
    manifest = Manifest(spec=suggest_spec(dataset))
    assert manifest.stack.is_empty
    assert manifest.render(dataset.frame).y_domain[0] == 0.0


def test_a_manifest_from_a_future_version_is_refused() -> None:
    payload = Manifest(spec=ChartSpec(family=ChartFamily.LINE, x="a", y="b")).to_dict()
    payload["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ManifestError, match="unsupported manifest version"):
        Manifest.from_dict(payload)


def test_a_malformed_manifest_is_refused() -> None:
    with pytest.raises(ManifestError, match="malformed"):
        Manifest.from_dict({"format_version": FORMAT_VERSION, "spec": {"family": "line"}})


def test_manifest_json_must_be_valid_json() -> None:
    with pytest.raises(ManifestError, match="not valid JSON"):
        Manifest.from_json("{not json")
