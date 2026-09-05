from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tidy_frame() -> pd.DataFrame:
    """A clean monthly series with a categorical split."""
    months = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m-%d")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "month": list(months) * 2,
            "revenue": np.round(rng.normal(1000, 50, 24), 2),
            "region": ["north"] * 12 + ["south"] * 12,
        }
    )


@pytest.fixture
def csv_bytes(tidy_frame: pd.DataFrame) -> bytes:
    return tidy_frame.to_csv(index=False).encode("utf-8")
