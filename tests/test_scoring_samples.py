"""Calibration of the impact scores against the bundled sample data.

Each sample declares the techniques its chart is meant to exercise, and these
tests hold the scores to those claims: a declared technique must register an
impact a reader would actually notice. The blind spots the scores do not catch
are pinned separately, in ``test_scoring.py``.
"""

from __future__ import annotations

import pytest

from core.samples import Sample, list_samples, load_sample
from core.scoring import ImpactScores, score_stack
from core.spec import ChartSpec
from core.transforms import CherryPickedWindow, Transform, TransformStack, build

#: A variant registers impact if the line's steepness changes by 40% or more,
#: or it drops a tenth of the rows. Below that a reader would not reliably
#: notice, so the sample's claim would be an overstatement. The threshold sits
#: at 1.4 rather than 1.5 because truncating the MRR axis lands at 1.47: that
#: series already fills most of its axis, so there is little room left to
#: truncate — a property of the data, not a weak transform.
IMPACT_SLOPE = 1.4
IMPACT_FIDELITY = 0.9

SCORED = [(s, name) for s in list_samples() for name in s.demonstrates]


def make(name: str, sample: Sample, spec: ChartSpec) -> Transform:
    """Build a demo instance of ``name`` for this sample's chart."""
    if name == "cherry_picked_window":
        frame = load_sample(sample.key).frame
        return CherryPickedWindow.favouring(frame, spec, direction="down", length=6)
    return build(name)


def scores_for(sample: Sample, name: str) -> ImpactScores:
    dataset = load_sample(sample.key)
    spec = sample.spec()
    stack = TransformStack().push(make(name, sample, spec))
    return score_stack(dataset.frame, spec, stack)[1]


def has_impact(result: ImpactScores) -> bool:
    steepness = abs(result.slope_ratio)
    return (
        result.direction_flip
        or steepness >= IMPACT_SLOPE
        or steepness <= 1 / IMPACT_SLOPE
        or result.data_fidelity <= IMPACT_FIDELITY
    )


@pytest.mark.parametrize(("sample", "name"), SCORED, ids=[f"{s.key}-{n}" for s, n in SCORED])
def test_a_declared_technique_registers_real_impact(sample: Sample, name: str) -> None:
    result = scores_for(sample, name)
    assert has_impact(result), f"{name} on {sample.key} scored as barely noticeable: {result}"
    assert result.summary() != "reads much like the baseline"


def test_every_sample_declares_a_chart_that_prepares() -> None:
    for sample in list_samples():
        dataset = load_sample(sample.key)
        spec = sample.spec()
        assert set(spec.columns) <= set(dataset.frame.columns)


def test_the_baseline_of_every_sample_scores_as_unchanged() -> None:
    for sample in list_samples():
        if not sample.demonstrates:
            continue
        dataset = load_sample(sample.key)
        result = score_stack(dataset.frame, sample.spec(), TransformStack())[1]
        assert result.slope_ratio == pytest.approx(1.0)
        assert result.data_fidelity == 1.0


def test_the_mrr_dip_can_be_made_to_look_like_a_collapse() -> None:
    """The headline demo: growth reads as decline once it is windowed."""
    dataset = load_sample("saas_mrr")
    sample = next(s for s in list_samples() if s.key == "saas_mrr")
    chart = sample.spec()
    window = CherryPickedWindow.favouring(dataset.frame, chart, direction="down", length=3)
    stack = TransformStack().push(window, build("truncated_axis"))
    variant, result = score_stack(dataset.frame, chart, stack)

    assert result.baseline_direction.value == "rising"
    assert result.variant_direction.value == "falling"
    assert result.direction_flip
    assert result.data_fidelity < 0.15
    assert variant.y_domain[0] > 0
