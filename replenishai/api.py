"""FastAPI service (M3, FR-7..10). Serves results that scripts/run_experiment.py already produced.

Design (docs/04-sad.md, data flow step 5): the API NEVER trains at request time. On startup it
loads three files from artifacts/:
  - results.json           accuracy + cost comparison + data_source  -> /compare, /health
  - holdout_forecasts.csv  28-day forecasts of the latest origin      -> /forecast, /replenish (mu)
  - series_stats.csv       sigma and price per series                 -> /replenish (sigma)
If any file is missing the app refuses to start and says which script to run.

All numbers come from replenishai.service, which the LLM assistant (/chat) also uses.
Every response carries `data_source` ("synthetic" or "m5") so synthetic numbers can never be
mistaken for real retail results (NFR-5).

Run:  uvicorn replenishai.api:app --reload      then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import service
from .service import (BASELINE, DATA_NOTES, MODEL, REQUIRED_FILES, Artifacts,  # noqa: F401
                      UnknownSeries, load_artifacts)

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
ARTIFACTS_ENV = "REPLENISHAI_ARTIFACTS"          # override the folder (used by tests / Docker)


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
    lead_time_days: int = Field(ge=service.LEAD_MIN, le=service.LEAD_MAX,
                                description="Supplier lead time L (1-28 days)")
    service_level: float = Field(ge=service.SL_MIN, le=service.SL_MAX,
                                 description="Target cycle service level")
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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="One question in natural language")


class ChatToolCall(BaseModel):
    name: str
    input: dict
    ok: bool


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ChatToolCall] = Field(description="Every tool the model called, in order")
    data_source: str
    data_note: str
    served_by: str = Field(description="Model version that produced the answer")


# ---------------------------------------------------------------- the app

def create_app(artifacts_dir: str | Path | None = None, llm_client=None) -> FastAPI:
    """`llm_client` lets tests inject a fake Gemini client for /chat."""
    folder = Path(artifacts_dir or os.environ.get(ARTIFACTS_ENV) or ROOT / "artifacts")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.art = load_artifacts(folder)     # fails fast if the experiment was not run
        yield

    app = FastAPI(title="ReplenishAI", version="0.5.0", lifespan=lifespan,
                  description="Demand forecast + replenishment decision API. "
                              "Check `data_source` in every response.")

    def art(request: Request) -> Artifacts:
        return request.app.state.art

    def not_found(e: UnknownSeries) -> HTTPException:
        return HTTPException(404, str(e))           # BR-1: the message names the bad field

    @app.get("/health")
    def health(request: Request) -> dict:
        a = art(request)
        return {"status": "ok", **service.source_fields(a), "forecast_origin": a.origin,
                "n_series": int(a.forecasts["id"].nunique())}

    @app.get("/", include_in_schema=False)
    def index():
        """FR-10: the dashboard page, served by the same app (one service, no CORS)."""
        return FileResponse(STATIC / "index.html")

    @app.get("/series")
    def series(request: Request) -> dict:
        """Stores and items available in the artifacts (fills the dashboard's dropdowns)."""
        return service.series_data(art(request))

    @app.get("/forecast", response_model=ForecastResponse)
    def forecast(item_id: str, store_id: str, request: Request):
        """FR-7: the 28-day LightGBM forecast of one item in one store."""
        try:
            return service.forecast_data(art(request), item_id, store_id)
        except UnknownSeries as e:
            raise not_found(e)

    @app.post("/replenish", response_model=ReplenishResponse)
    def replenish(req: ReplenishRequest, request: Request):
        """FR-8: reorder point and order quantity (formula in docs/05-sds.md section 2)."""
        try:
            return service.replenish_data(art(request), req.item_id, req.store_id, req.on_hand,
                                          req.lead_time_days, req.service_level, req.moq)
        except UnknownSeries as e:
            raise not_found(e)

    @app.get("/compare")
    def compare(request: Request) -> dict:
        """FR-9: cost and accuracy of every strategy, copied from artifacts/results.json."""
        return service.compare_data(art(request))

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request):
        """FR-11: natural-language question -> Gemini calls the tools above -> answer.
        503 when no Gemini API key is configured; the rest of the API is unaffected."""
        from .assistant import AssistantUnavailable, ask     # imported lazily: optional feature
        a = art(request)
        try:
            res = await run_in_threadpool(ask, req.question, a, llm_client)
        except AssistantUnavailable as e:
            raise HTTPException(503, f"LLM 助理目前無法使用：{e}")
        except ValueError as e:
            raise HTTPException(422, str(e))
        return {"answer": res.answer, "served_by": res.served_by, **service.source_fields(a),
                "tool_calls": [{"name": c.name, "input": c.input, "ok": c.ok} for c in res.tool_calls]}

    return app


app = create_app()
