"""Access to the bundled demo datasets.

Each sample was built to exercise a particular family of distortions; see
``data/samples/generate.py`` for how they are produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.ingest import Dataset, load
from core.spec import ChartFamily, ChartSpec

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@dataclass(frozen=True, slots=True)
class Sample:
    """A bundled dataset, the chart it was built around, and what it shows.

    The chart matters: a technique only distorts the reading of a particular
    chart, so ``demonstrates`` is a claim about :meth:`spec`, not about the
    file in general. Swapping the measure or splitting by a different column
    can turn any of these techniques into a no-op.
    """

    key: str
    filename: str
    title: str
    description: str
    #: Techniques whose distortion of this chart the impact scores capture.
    demonstrates: tuple[str, ...]
    family: ChartFamily
    x: str
    y: str | None = None
    series: str | None = None

    @property
    def path(self) -> Path:
        return SAMPLES_DIR / self.filename

    def spec(self) -> ChartSpec:
        """The baseline chart this sample's claims are calibrated against."""
        return ChartSpec(family=self.family, x=self.x, y=self.y, series=self.series)


SAMPLES: dict[str, Sample] = {
    sample.key: sample
    for sample in (
        Sample(
            key="saas_mrr",
            filename="saas_mrr.csv",
            title="SaaS monthly recurring revenue",
            description="36 months of MRR with strong growth and one bad quarter.",
            demonstrates=("truncated_axis", "cherry_picked_window", "cumulative"),
            family=ChartFamily.LINE,
            x="month",
            y="mrr_usd",
        ),
        Sample(
            key="support_tickets",
            filename="support_tickets.csv",
            title="Daily support ticket volume",
            description="Two noisy years with weekly seasonality and a flat underlying trend.",
            demonstrates=("smoothing", "cherry_picked_window", "outlier_drop"),
            family=ChartFamily.LINE,
            x="date",
            y="tickets",
        ),
        Sample(
            key="regional_sales",
            filename="regional_sales.csv",
            title="Quarterly sales by region",
            description="Eight regions with a heavy long tail across eight quarters.",
            # Split by region, so regrouping genuinely changes which lines a
            # reader compares. Truncation is not on this list: the long tail
            # reaches near zero, so there is no room to raise the floor.
            demonstrates=("ratio_vs_absolute", "bin_regrouping"),
            family=ChartFamily.BAR,
            x="quarter",
            y="revenue_usd",
            series="region",
        ),
        Sample(
            key="messy_inventory",
            filename="messy_inventory.csv",
            title="Inventory export (dirty)",
            description=(
                "A deliberately dirty table: duplicate header, empty column, "
                "mixed number/text column, sparse column."
            ),
            demonstrates=(),
            family=ChartFamily.BAR,
            x="warehouse",
            y="quantity",
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
