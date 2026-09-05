"""Generate the bundled sample datasets.

Each sample is built to exercise a specific family of distortions, and each is
fully deterministic: rerunning this script reproduces the committed CSVs byte
for byte. Run it with ``python data/samples/generate.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEED = 20240101


def saas_mrr() -> pd.DataFrame:
    """36 months of MRR: strong growth with one bad quarter.

    Target distortions: truncated axis (the dip looks catastrophic),
    cherry-picked window (the bad quarter alone reads as collapse), cumulative
    (hides the dip entirely).
    """
    rng = np.random.default_rng(SEED)
    months = pd.date_range("2022-01-01", periods=36, freq="MS")
    base = 42_000 * 1.031 ** np.arange(36)
    base[18:21] *= [0.94, 0.88, 0.91]
    mrr = np.round(base * rng.normal(1.0, 0.012, 36), 2)
    churn = np.round(
        rng.normal(0.031, 0.004, 36) + np.r_[np.zeros(18), 0.02, 0.03, 0.02, np.zeros(15)], 4
    )
    return pd.DataFrame(
        {
            "month": months.strftime("%Y-%m-%d"),
            "mrr_usd": mrr,
            "customers": np.round(mrr / rng.normal(310, 8, 36)).astype(int),
            "churn_rate": churn,
        }
    )


def support_tickets() -> pd.DataFrame:
    """Two years of daily ticket volume: noisy, weekly seasonality, flat trend.

    Target distortions: smoothing (invents a clean trend), aggregation swap
    (mean vs median diverge on spike days), cherry-picked window (any 30-day
    slice can be made to rise or fall).
    """
    rng = np.random.default_rng(SEED + 1)
    days = pd.date_range("2023-01-01", periods=730, freq="D")
    weekday_effect = np.where(days.dayofweek >= 5, 0.45, 1.0)
    spikes = rng.random(730) < 0.03
    volume = rng.poisson(120, 730) * weekday_effect + spikes * rng.integers(80, 400, 730)
    return pd.DataFrame(
        {
            "date": days.strftime("%Y-%m-%d"),
            "tickets": volume.round().astype(int),
            "channel": rng.choice(["email", "chat", "phone"], 730, p=[0.5, 0.35, 0.15]),
            "first_response_mins": np.round(rng.gamma(2.0, 18.0, 730), 1),
        }
    )


def regional_sales() -> pd.DataFrame:
    """Sales by region and quarter with a heavy long tail.

    Target distortions: bin regrouping (fold the tail into "Other" and the
    leader dominates), aggregation swap (mean is dragged by two outliers,
    median is not), ratio vs absolute (small regions post huge % growth).
    """
    rng = np.random.default_rng(SEED + 2)
    regions = ["North", "South", "East", "West", "Nordics", "Iberia", "Benelux", "Baltics"]
    weights = [1.0, 0.85, 0.7, 0.62, 0.14, 0.11, 0.08, 0.03]
    quarters = [f"{y}-Q{q}" for y in (2023, 2024) for q in (1, 2, 3, 4)]
    rows = []
    for region, weight in zip(regions, weights, strict=True):
        for i, quarter in enumerate(quarters):
            revenue = 250_000 * weight * (1.04**i) * rng.normal(1.0, 0.06)
            rows.append(
                {
                    "quarter": quarter,
                    "region": region,
                    "revenue_usd": round(revenue, 2),
                    "units": int(revenue / rng.normal(85, 6)),
                }
            )
    return pd.DataFrame(rows)


def messy_inventory() -> pd.DataFrame:
    """A deliberately dirty table for the validation path.

    Contains a duplicate header, an all-null column, a column mixing numbers
    and text, and a sparse column. Nothing here is meant to chart well.
    """
    rng = np.random.default_rng(SEED + 3)
    n = 60
    quantity = [str(int(v)) for v in rng.integers(0, 400, n)]
    for i in range(0, n, 7):
        quantity[i] = "unknown"
    reorder = [str(round(float(v), 1)) for v in rng.normal(50, 12, n)]
    for i in range(0, n, 3):
        reorder[i] = ""
    return pd.DataFrame(
        {
            "sku": [f"SKU-{i:04d}" for i in range(n)],
            "warehouse": rng.choice(["ams", "dub", "sfo"], n),
            "quantity": quantity,
            "notes": [None] * n,
            "reorder_point": reorder,
            "last_counted": pd.date_range("2024-06-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        }
    )


SAMPLES = {
    "saas_mrr.csv": saas_mrr,
    "support_tickets.csv": support_tickets,
    "regional_sales.csv": regional_sales,
    "messy_inventory.csv": messy_inventory,
}


def main() -> None:
    for filename, build in SAMPLES.items():
        path = HERE / filename
        build().to_csv(path, index=False, lineterminator="\n")
        print(f"wrote {path.relative_to(HERE.parents[1])}")
    # messy_inventory needs a genuinely duplicated header, which pandas will
    # not write, so duplicate the warehouse column in place afterwards.
    path = HERE / "messy_inventory.csv"
    lines = path.read_text().split("\n")
    patched = []
    for line in lines:
        if not line:
            patched.append(line)
            continue
        fields = line.split(",")
        fields.insert(2, fields[1])
        patched.append(",".join(fields))
    path.write_text("\n".join(patched))
    print(f"patched duplicate header into {path.name}")


if __name__ == "__main__":
    main()
