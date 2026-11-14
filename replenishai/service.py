"""Read-only queries over the experiment artifacts, shared by the REST API and the LLM assistant.

Both callers MUST go through these functions so the assistant can never report a number that the
API (and the dashboard) would not report for the same inputs. Nothing here trains a model.

Errors are plain Python exceptions; each caller maps them to its own surface:
  UnknownSeries (LookupError) -> HTTP 404 / a tool error the model must relay
  ValueError                   -> HTTP 422 / a tool error (bad input, e.g. lead time 40)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import HORIZON
from .policy import order_quantity, reorder_point, z_from_service_level

REQUIRED_FILES = ("results.json", "holdout_forecasts.csv", "series_stats.csv")
MODEL = "lightgbm"                               # the model whose forecast is served
BASELINE = "moving_avg_28"                       # what the cost comparison is measured against
DATA_NOTES = {
    "synthetic": "合成資料（synthetic）產生的結果，不代表真實零售表現。",
    "m5": "資料來源：M5（Walmart）公開資料集。",
}
COST_PARAMS_NOTE = "假設值（待確認）：M5 沒有真實成本資料。"
# BR-4 limits, also enforced by the API's pydantic model
LEAD_MIN, LEAD_MAX = 1, HORIZON
SL_MIN, SL_MAX = 0.5, 0.999


class UnknownSeries(LookupError):
    """Item or store not in the artifacts (BR-1). The message names the bad field."""


@dataclass
class Artifacts:
    results: dict
    forecasts: pd.DataFrame       # one row per (series, day) of the latest 28-day window
    stats: pd.DataFrame           # index = series id; columns sigma, price

    @property
    def data_source(self) -> str:
        return self.results["data_source"]

    @property
    def origin(self) -> str:
        """Forecast origin = last day of training data; forecasts start the day after."""
        return self.results["origins"][-1]


def load_artifacts(folder: str | Path) -> Artifacts:
    folder = Path(folder)
    missing = [f for f in REQUIRED_FILES if not (folder / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing artifacts in {folder}: {', '.join(missing)}. "
            "Run `python scripts/run_experiment.py` first to produce them.")
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    forecasts = pd.read_csv(folder / "holdout_forecasts.csv").sort_values(["id", "date"])
    stats = pd.read_csv(folder / "series_stats.csv").set_index("id")
    return Artifacts(results=results, forecasts=forecasts, stats=stats)


def source_fields(a: Artifacts) -> dict:
    return {"data_source": a.data_source, "data_note": DATA_NOTES.get(a.data_source, "")}


def series_rows(a: Artifacts, item_id: str, store_id: str) -> pd.DataFrame:
    f = a.forecasts
    if item_id not in set(f["item_id"]):
        raise UnknownSeries(f"item_id not found: {item_id}")
    if store_id not in set(f["store_id"]):
        raise UnknownSeries(f"store_id not found: {store_id}")
    rows = f[(f["item_id"] == item_id) & (f["store_id"] == store_id)]
    if rows.empty:
        raise UnknownSeries(f"no series for item_id={item_id}, store_id={store_id}")
    return rows


def series_data(a: Artifacts) -> dict:
    pairs = a.forecasts[["store_id", "item_id"]].drop_duplicates()
    return {**source_fields(a),
            "stores": sorted(pairs["store_id"].unique().tolist()),
            "items_by_store": {s: sorted(g["item_id"].tolist())
                               for s, g in pairs.groupby("store_id")}}


def forecast_data(a: Artifacts, item_id: str, store_id: str) -> dict:
    rows = series_rows(a, item_id, store_id)
    points = [{"date": d, "pred": round(float(p), 4), "actual": float(y)}
              for d, p, y in zip(rows["date"], rows[f"pred_{MODEL}"], rows["sales"])]
    return {"item_id": item_id, "store_id": store_id, **source_fields(a), "model": MODEL,
            "forecast_origin": a.origin, "horizon_days": len(points), "forecast": points}


def replenish_data(a: Artifacts, item_id: str, store_id: str, on_hand: int,
                   lead_time_days: int, service_level: float, moq: int = 0) -> dict:
    """Reorder point and order quantity (formula in docs/05-sds.md section 2)."""
    if on_hand < 0:
        raise ValueError("on_hand must be >= 0")
    if not LEAD_MIN <= lead_time_days <= LEAD_MAX:
        raise ValueError(f"lead_time_days must be between {LEAD_MIN} and {LEAD_MAX}")
    if not SL_MIN <= service_level <= SL_MAX:
        raise ValueError(f"service_level must be between {SL_MIN} and {SL_MAX}")
    if moq < 0:
        raise ValueError("moq must be >= 0")
    rows = series_rows(a, item_id, store_id)
    L = lead_time_days
    mu = float(rows[f"pred_{MODEL}"].iloc[:L].mean())          # forecast over the next L days
    sigma = float(a.stats.loc[rows["id"].iloc[0], "sigma"])
    z = z_from_service_level(service_level)
    rop = reorder_point(mu, sigma, L, z)
    position = float(on_hand)                                    # on-order = 0 in the MVP
    return {
        "item_id": item_id, "store_id": store_id, **source_fields(a),
        "reorder_point": round(rop, 3), "safety_stock": round(rop - mu * L, 3),
        "order_qty": order_quantity(position, rop, moq),
        "mu_daily": round(mu, 4), "sigma_daily": round(sigma, 4), "z": round(z, 4),
        "inventory_position": position,
    }


def compare_data(a: Artifacts) -> dict:
    r = a.results
    strategies = [{"name": name, **cost, **r["accuracy"].get(name, {})}
                  for name, cost in r["cost"].items()]
    return {
        **source_fields(a), "horizon_days": HORIZON, "origins": r["origins"],
        "lead_time_days": r["lead_time"], "service_level": r["service_level"],
        "cost_params": r.get("cost_params"), "cost_params_note": COST_PARAMS_NOTE,
        "baseline": BASELINE, "strategies": strategies,
        "sensitivity": r.get("sensitivity", []),
        "bootstrap_vs_baseline": r.get("bootstrap_vs_ma28", {}),
    }
