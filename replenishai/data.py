"""Load M5-format files into one tidy long table (one row = one series on one day)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HORIZON = 28  # forecast horizon in days (same as the M5 competition)


def load_m5(
    raw_dir: str | Path,
    stores: list[str] | None = None,
    max_items: int | None = None,
) -> pd.DataFrame:
    """Return columns: id, item_id, dept_id, cat_id, store_id, date, sales, sell_price,
    event, snap, wday, month, year. Works for real M5 and the synthetic copy.

    `stores` / `max_items` shrink the data BEFORE the wide->long melt (real M5 has 30k series).
    """
    raw = Path(raw_dir)
    cal = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(raw / "sell_prices.csv")
    wide = pd.read_csv(raw / "sales_train_evaluation.csv")

    if stores is not None:
        wide = wide[wide["store_id"].isin(stores)]
    if max_items is not None:
        keep = sorted(wide["item_id"].unique())[:max_items]
        wide = wide[wide["item_id"].isin(keep)]

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    long = wide.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")

    # snap flag of the series' own state
    snap_long = cal.melt(id_vars=["d"], value_vars=[c for c in cal.columns if c.startswith("snap_")],
                         var_name="snap_col", value_name="snap")
    snap_long["state_id"] = snap_long["snap_col"].str.replace("snap_", "", regex=False)

    cal_keep = cal[["d", "date", "wm_yr_wk", "wday", "month", "year", "event_name_1"]]
    long = (long.merge(cal_keep, on="d", how="left")
                .merge(snap_long[["d", "state_id", "snap"]], on=["d", "state_id"], how="left")
                .merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"))

    # No price = item was not on sale yet (real M5 behaviour): drop those rows.
    long = long.dropna(subset=["sell_price"])
    long["event"] = long["event_name_1"].notna().astype(int)
    long = long.drop(columns=["d", "wm_yr_wk", "event_name_1", "state_id"])
    long = long.sort_values(["id", "date"]).reset_index(drop=True)
    long["sales"] = long["sales"].astype(float)
    return long


def split_train_test(df: pd.DataFrame, horizon: int = HORIZON):
    """Hold out the last `horizon` days. Returns (train, test, origin_date)."""
    last = df["date"].max()
    origin = last - pd.Timedelta(days=horizon)
    return df[df["date"] <= origin], df[df["date"] > origin], origin
