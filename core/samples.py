"""Access to the bundled demo datasets.

Each sample was built to exercise a particular family of distortions; see
``data/samples/generate.py`` for how they are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.ingest import Dataset, load

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@dataclass(frozen=True, slots=True)
class Sample:
    """A bundled dataset and what it is good for demonstrating."""

    key: str
    filename: str
    title: str
    description: str
    demonstrates: tuple[str, ...]

    @property
    def path(self) -> Path:
        return SAMPLES_DIR / self.filename


SAMPLES: dict[str, Sample] = {
    sample.key: sample
    for sample in (
        Sample(
            key="saas_mrr",
            filename="saas_mrr.csv",
            title="SaaS monthly recurring revenue",
            description="36 months of MRR with strong growth and one bad quarter.",
            demonstrates=("truncated_axis", "cherry_picked_window", "cumulative"),
        ),
        Sample(
            key="support_tickets",
            filename="support_tickets.csv",
            title="Daily support ticket volume",
            description="Two noisy years with weekly seasonality and a flat underlying trend.",
            demonstrates=("smoothing", "aggregation_swap", "cherry_picked_window"),
        ),
        Sample(
            key="regional_sales",
            filename="regional_sales.csv",
            title="Quarterly sales by region",
            description="Eight regions with a heavy long tail across eight quarters.",
            demonstrates=("bin_regrouping", "aggregation_swap", "ratio_vs_absolute"),
        ),
        Sample(
            key="messy_inventory",
            filename="messy_inventory.csv",
            title="Inventory export (dirty)",
            description=(
                "A deliberately dirty table: duplicate header, empty column, "
                "mixed number/text column, sparse column."
            ),
            demonstrates=("validation",),
        ),
    )
}


def list_samples() -> tuple[Sample, ...]:
    """Every bundled sample, in menu order."""
    return tuple(SAMPLES.values())


def load_sample(key: str) -> Dataset:
    """Load a bundled sample by key.

    Raises:
        KeyError: No sample by that name.
    """
    sample = SAMPLES[key]
    return load(sample.path)
