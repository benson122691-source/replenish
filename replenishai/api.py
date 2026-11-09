"""FastAPI service (M3, FR-7..9). Serves results that scripts/run_experiment.py already produced.

Design (docs/04-sad.md, data flow step 5): the API NEVER trains at request time. On startup it
loads three files from artifacts/:
  - results.json           accuracy + cost comparison + data_source  -> /compare, /health
  - holdout_forecasts.csv  28-day forecasts of the latest origin      -> /forecast, /replenish (mu)
  - series_stats.csv       sigma and price per series                 -> /replenish (sigma)
If any file is missing the app refuses to start and says which script to run.

Every response carries `data_source` ("synthetic" or "m5") so synthetic numbers can never be
mistaken for real retail results (NFR-5).

Run:  uvicorn replenishai.api:app --reload      then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .data import HORIZON
from .policy import order_quantity, reorder_point, z_from_service_level

ROOT = Path(__file__).resolve().parent.parent
STATIC_INDEX = ROOT / "static" / "index.html"
ARTIFACTS_ENV = "REPLENISHAI_ARTIFACTS"          # override the folder (used by tests / Docker)
REQUIRED_FILES = ("results.json", "holdout_forecasts.csv", "series_stats.csv")
MODEL = "lightgbm"                               # the model whose forecast the API serves
BASELINE = "moving_avg_28"                       # what the cost comparison is measured against
DATA_NOTES = {
    "synthetic": "合成資料（synthetic）產生的結果，不代表真實零售表現。",
    "m5": "資料來源：M5（Walmart）公開資料集。",
}


# ---------------------------------------------------------------- loading artifacts

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


# ---------------------------------------------------------------- request / response schemas

class ForecastPoint(BaseModel):
    date: str
    pred: float = Field(description="LightGBM forecast (units/day), clipped at 0")
    actual: float = Field(description="Real demand of that day in the hold-out window (backtest)")


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    data_source: str
    data_note: str
    model: str
    forecast_origin: str
    horizon_days: int
    forecast: list[ForecastPoint]


class ReplenishRequest(BaseModel):
    """Field limits implement the boundary rule BR-4; FastAPI answers 422 when one is broken."""
    item_id: str
    store_id: str
    on_hand: int = Field(ge=0, description="Units in stock now")
    lead_time_days: int = Field(ge=1, le=HORIZON, description="Supplier lead time L (1-28 days)")
    service_level: float = Field(ge=0.5, le=0.999, description="Target cycle service level")
    moq: int = Field(default=0, ge=0, description="Minimum order quantity (0 = none)")


class ReplenishResponse(BaseModel):
    item_id: str
    store_id: str
    data_source: str
    data_note: str
    reorder_point: float = Field(description="ROP = mu*L + z*sigma*sqrt(L)")
    safety_stock: float = Field(description="z*sigma*sqrt(L)")
    order_qty: int = Field(description="0 if on_hand > ROP, else max(MOQ, ceil(ROP - on_hand))")
    mu_daily: float = Field(description="Mean LightGBM forecast over the next L days")
    sigma_daily: float = Field(description="Std of daily sales, last 56 training days")
    z: float
    inventory_position: float = Field(description="on_hand + on_order (on_order = 0 in the MVP)")


# ---------------------------------------------------------------- the app

def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    folder = Path(artifacts_dir or os.environ.get(ARTIFACTS_ENV) or ROOT / "artifacts")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.art = load_artifacts(folder)     # fails fast if the experiment was not run
        yield

    app = FastAPI(title="ReplenishAI", version="0.3.0", lifespan=lifespan,
                  description="Demand forecast + replenishment decision API. "
                              "Check `data_source` in every response.")

    def art(request: Request) -> Artifacts:
        return request.app.state.art

    def series_rows(a: Artifacts, item_id: str, store_id: str) -> pd.DataFrame:
        """BR-1: unknown item or store -> 404 naming the bad field."""
        f = a.forecasts
        if item_id not in set(f["item_id"]):
            raise HTTPException(404, f"item_id not found: {item_id}")
        if store_id not in set(f["store_id"]):
            raise HTTPException(404, f"store_id not found: {store_id}")
        rows = f[(f["item_id"] == item_id) & (f["store_id"] == store_id)]
        if rows.empty:
            raise HTTPException(404, f"no series for item_id={item_id}, store_id={store_id}")
        return rows

    def source_fields(a: Artifacts) -> dict:
        return {"data_source": a.data_source, "data_note": DATA_NOTES.get(a.data_source, "")}

    @app.get("/health")
    def health(request: Request) -> dict:
        a = art(request)
        return {"status": "ok", **source_fields(a), "forecast_origin": a.origin,
                "n_series": int(a.forecasts["id"].nunique())}

    @app.get("/", include_in_schema=False)
    def index():
        """The one-page dashboard (static/index.html); it only calls the JSON endpoints below."""
        return FileResponse(STATIC_INDEX)

    @app.get("/series")
    def series(request: Request) -> dict:
        """Items and stores that exist in the served forecast (fills the dashboard dropdowns)."""
        f = art(request).forecasts
        pairs = f[["item_id", "store_id"]].drop_duplicates()
        return {"stores": sorted(pairs["store_id"].unique()),
                "items_by_store": {st: sorted(g["item_id"]) for st, g in pairs.groupby("store_id")}}

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(item_id: str, store_id: str, request: Request):
        """FR-7: the 28-day LightGBM forecast of one item in one store."""
        a = art(request)
        rows = series_rows(a, item_id, store_id)
        points = [{"date": d, "pred": round(float(p), 4), "actual": float(y)}
                  for d, p, y in zip(rows["date"], rows[f"pred_{MODEL}"], rows["sales"])]
        return {"item_id": item_id, "store_id": store_id, **source_fields(a), "model": MODEL,
                "forecast_origin": a.origin, "horizon_days": len(points), "forecast": points}

    @app.post("/replenish", response_model=ReplenishResponse)
    def replenish(req: ReplenishRequest, request: Request):
        """FR-8: reorder point and order quantity (formula in docs/05-sds.md section 2)."""
        a = art(request)
        rows = series_rows(a, req.item_id, req.store_id)
        L = req.lead_time_days
        mu = float(rows[f"pred_{MODEL}"].iloc[:L].mean())          # forecast over the next L days
        sigma = float(a.stats.loc[rows["id"].iloc[0], "sigma"])
        z = z_from_service_level(req.service_level)
        rop = reorder_point(mu, sigma, L, z)
        position = float(req.on_hand)                                # on-order = 0 in the MVP
        return {
            "item_id": req.item_id, "store_id": req.store_id, **source_fields(a),
            "reorder_point": round(rop, 3), "safety_stock": round(rop - mu * L, 3),
            "order_qty": order_quantity(position, rop, req.moq),
            "mu_daily": round(mu, 4), "sigma_daily": round(sigma, 4), "z": round(z, 4),
            "inventory_position": position,
        }

    @app.get("/compare")
    def compare(request: Request) -> dict:
        """FR-9: cost and accuracy of every strategy, copied from artifacts/results.json."""
        a = art(request)
        r = a.results
        strategies = [{"name": name, **cost, **r["accuracy"].get(name, {})}
                      for name, cost in r["cost"].items()]
        return {
            **source_fields(a), "horizon_days": HORIZON, "origins": r["origins"],
            "lead_time_days": r["lead_time"], "service_level": r["service_level"],
            "cost_params": r.get("cost_params"),
            "cost_params_note": "假設值（待確認）：M5 沒有真實成本資料。",
            "baseline": BASELINE, "strategies": strategies,
            "sensitivity": r.get("sensitivity", []),
            "bootstrap_vs_baseline": r.get("bootstrap_vs_ma28", {}),
        }

    return app


app = create_app()
