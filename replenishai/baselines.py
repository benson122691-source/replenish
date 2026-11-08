"""Baseline forecasters and error metrics. Any fancy model must beat these."""
from __future__ import annotations

import numpy as np
import pandas as pd


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, period: int = 7) -> pd.DataFrame:
    """Predict each future day with the most recent same-weekday value we already know."""
    origin = train["date"].max()
    out = []
    for sid, g in train.groupby("id"):
        s = g.set_index("date")["sales"]
        t = test[test["id"] == sid]
        preds = []
        for d in t["date"]:
            k = int(np.ceil((d - origin).days / period))   # weeks to step back
            ref = d - pd.Timedelta(days=period * k)
            preds.append(s.get(ref, s.tail(period).mean()))
        out.append(pd.DataFrame({"id": sid, "date": t["date"].values, "pred": preds}))
    return pd.concat(out, ignore_index=True)


def moving_average(train: pd.DataFrame, test: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """Flat forecast = mean of the last `window` days (what many shops do in Excel)."""
    mean = (train.sort_values("date").groupby("id")["sales"]
                 .apply(lambda s: s.tail(window).mean()).rename("pred").reset_index())
    return test[["id", "date"]].merge(mean, on="id", how="left")


def score(test: pd.DataFrame, pred: pd.DataFrame, train: pd.DataFrame) -> dict:
    """RMSE / MAE / bias on the hold-out, plus RMSSE (error scaled by naive one-step error,
    the M5-style scale-free metric: <1 means better than 'tomorrow = today')."""
    m = test[["id", "date", "sales"]].merge(pred, on=["id", "date"], how="left")
    err = m["pred"] - m["sales"]

    scale = (train.sort_values("date").groupby("id")["sales"]
                  .apply(lambda s: np.mean(np.diff(s.to_numpy()) ** 2)))
    mse_by_id = (err ** 2).groupby(m["id"]).mean()
    rmsse = np.sqrt(mse_by_id / scale.replace(0, np.nan)).dropna()
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "rmsse": float(rmsse.mean()),
    }
