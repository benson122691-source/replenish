"""Feature engineering for a direct 28-day-ahead forecast.

KEY IDEA (no leakage): every lag/rolling feature is shifted by at least HORIZON (28) days.
So at forecast origin T, a target day T+h (h = 1..28) only needs sales up to T+h-28 <= T.
Prices, events and calendar are known in advance, so they can be used as-is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HORIZON

CATEGORICAL = ["item_code", "store_code", "dept_code"]
NUMERIC = [
    "lag_28", "lag_35", "lag_42", "lag_56",
    "roll_mean_7", "roll_mean_28", "roll_std_28",
    "sell_price", "price_ratio",
    "wday", "month", "dayofmonth", "weekofyear", "event", "snap",
]
FEATURES = CATEGORICAL + NUMERIC


def add_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id", sort=False)["sales"]

    for lag in (28, 35, 42, 56):
        df[f"lag_{lag}"] = g.shift(lag)

    base = g.shift(horizon)  # sales known `horizon` days before the target day
    grp = base.groupby(df["id"], sort=False)
    df["roll_mean_7"] = grp.transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["roll_mean_28"] = grp.transform(lambda s: s.rolling(28, min_periods=7).mean())
    df["roll_std_28"] = grp.transform(lambda s: s.rolling(28, min_periods=7).std())

    # price is known in advance: compare to the recent highest price (promotion detector)
    pmax = df.groupby("id", sort=False)["sell_price"].transform(
        lambda s: s.rolling(91, min_periods=1).max())
    df["price_ratio"] = df["sell_price"] / pmax

    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    for col, new in (("item_id", "item_code"), ("store_id", "store_code"), ("dept_id", "dept_code")):
        df[new] = df[col].astype("category").cat.codes
    return df


def feature_columns() -> list[str]:
    return list(FEATURES)


def assert_no_future_leak(df: pd.DataFrame, horizon: int = HORIZON) -> None:
    """Cheap sanity check used in tests: lag_28 must equal sales 28 rows (days) earlier."""
    one = df[df["id"] == df["id"].iloc[0]].reset_index(drop=True)
    expected = one["sales"].shift(horizon)
    ok = np.allclose(one["lag_28"].fillna(-1), expected.fillna(-1))
    if not ok:
        raise AssertionError("lag_28 does not match sales shifted by the horizon")
