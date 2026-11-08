"""API tests (M3). TC numbers refer to docs/06-test-plan.md.

TestClient sends fake HTTP requests to the app in-process (no real server needed).
The tests read the REAL artifacts written by scripts/run_experiment.py and check that the API
returns exactly those numbers: the API must never invent or alter results.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from replenishai.api import REQUIRED_FILES, create_app
from replenishai.policy import order_quantity, reorder_point, z_from_service_level

ART = Path(__file__).resolve().parent.parent / "artifacts"
HAVE_ARTIFACTS = all((ART / f).exists() for f in REQUIRED_FILES)
needs_artifacts = pytest.mark.skipif(
    not HAVE_ARTIFACTS, reason="artifacts/ missing: run `python scripts/run_experiment.py` first")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ART)) as c:     # `with` runs startup, i.e. loads the artifacts
        yield c


@pytest.fixture(scope="module")
def results() -> dict:
    return json.loads((ART / "results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def forecasts() -> pd.DataFrame:
    return pd.read_csv(ART / "holdout_forecasts.csv").sort_values(["id", "date"])


@pytest.fixture(scope="module")
def series(forecasts) -> dict:
    """A real (item, store) pair taken from the artifacts, not hard-coded."""
    row = forecasts.iloc[0]
    return {"item_id": row["item_id"], "store_id": row["store_id"], "id": row["id"]}


def replenish_body(series, **over) -> dict:
    body = {"item_id": series["item_id"], "store_id": series["store_id"], "on_hand": 10,
            "lead_time_days": 7, "service_level": 0.95, "moq": 0}
    return {**body, **over}


# ---------------------------------------------------------------- /health

@needs_artifacts
def test_health_reports_data_source(client, results):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["data_source"] == results["data_source"]


# ---------------------------------------------------------------- /forecast

@needs_artifacts
def test_forecast_returns_28_days_from_artifacts(client, series, forecasts, results):  # TC-13
    r = client.get("/forecast", params={"item_id": series["item_id"], "store_id": series["store_id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["data_source"] == results["data_source"]                 # NFR-5
    assert body["horizon_days"] == 28 and len(body["forecast"]) == 28    # AC-1.1
    assert all(p["pred"] >= 0 for p in body["forecast"])                 # BR-2

    expected = forecasts[forecasts["id"] == series["id"]]
    got = [p["pred"] for p in body["forecast"]]
    assert got == pytest.approx(expected["pred_lightgbm"].tolist(), abs=1e-4)
    assert [p["date"] for p in body["forecast"]] == expected["date"].tolist()


@needs_artifacts
@pytest.mark.parametrize("field,params", [
    ("item_id", {"item_id": "NOPE_999", "store_id": "CA_1"}),
    ("store_id", {"item_id": None, "store_id": "XX_9"}),
])
def test_forecast_unknown_item_or_store_is_404(client, series, field, params):  # TC-14, BR-1
    params = {**params, "item_id": params["item_id"] or series["item_id"]}
    r = client.get("/forecast", params=params)
    assert r.status_code == 404
    assert field in r.json()["detail"]


@needs_artifacts
def test_forecast_missing_query_param_is_422(client, series):
    assert client.get("/forecast", params={"item_id": series["item_id"]}).status_code == 422


# ---------------------------------------------------------------- /replenish

@needs_artifacts
@pytest.mark.parametrize("bad", [
    {"service_level": 1.2}, {"service_level": 0.4}, {"on_hand": -5},
    {"lead_time_days": 0}, {"lead_time_days": 29}, {"moq": -1},
])
def test_replenish_rejects_invalid_input(client, series, bad):  # TC-15, BR-4, AC-2.3
    r = client.post("/replenish", json=replenish_body(series, **bad))
    assert r.status_code == 422
    assert list(bad)[0] in json.dumps(r.json()["detail"])          # error names the field


@needs_artifacts
def test_replenish_matches_policy_formula(client, series, forecasts):  # TC-16, AC-2.1
    """Recompute ROP and order qty independently from the artifact files and compare."""
    r = client.post("/replenish", json=replenish_body(series, on_hand=3, lead_time_days=7))
    assert r.status_code == 200
    body = r.json()

    stats = pd.read_csv(ART / "series_stats.csv").set_index("id")
    mu = forecasts[forecasts["id"] == series["id"]]["pred_lightgbm"].iloc[:7].mean()
    sigma = stats.loc[series["id"], "sigma"]
    z = z_from_service_level(0.95)
    rop = reorder_point(mu, sigma, 7, z)

    assert body["reorder_point"] == pytest.approx(rop, abs=1e-3)
    assert body["sigma_daily"] == pytest.approx(sigma, abs=1e-4)
    assert body["order_qty"] == order_quantity(3, rop, 0)
    assert isinstance(body["order_qty"], int) and body["order_qty"] >= 0     # BR-3
    assert body["safety_stock"] == pytest.approx(z * sigma * math.sqrt(7), abs=1e-3)


@needs_artifacts
def test_replenish_zero_when_stock_is_enough(client, series):  # BR-3
    r = client.post("/replenish", json=replenish_body(series, on_hand=10**6))
    assert r.status_code == 200 and r.json()["order_qty"] == 0


@needs_artifacts
def test_replenish_respects_moq(client, series):  # AC-2.2
    r = client.post("/replenish", json=replenish_body(series, on_hand=0, moq=500))
    body = r.json()
    assert body["reorder_point"] < 500          # so the MOQ, not the gap, decides the quantity
    assert body["order_qty"] == 500


@needs_artifacts
def test_replenish_unknown_item_is_404(client, series):  # BR-1
    r = client.post("/replenish", json=replenish_body(series, item_id="NOPE_999"))
    assert r.status_code == 404 and "item_id" in r.json()["detail"]


# ---------------------------------------------------------------- /compare

@needs_artifacts
def test_compare_returns_artifact_numbers_and_data_source(client, results):  # TC-17, AC-3.1/3.2
    r = client.get("/compare")
    assert r.status_code == 200
    body = r.json()
    assert body["data_source"] == results["data_source"]
    assert body["horizon_days"] == 28
    assert body["cost_params"] == results.get("cost_params")

    by_name = {s["name"]: s for s in body["strategies"]}
    assert {"moving_avg_28", "lightgbm"} <= set(by_name)
    for name, cost in results["cost"].items():                  # every number = artifacts
        for k, v in {**cost, **results["accuracy"][name]}.items():
            assert by_name[name][k] == v, f"{name}.{k}"


@needs_artifacts
def test_synthetic_results_are_labelled(client, results):  # NFR-5
    if results["data_source"] != "synthetic":
        pytest.skip("artifacts come from real M5 data")
    for path in ("/health", "/compare"):
        body = client.get(path).json()
        assert body["data_source"] == "synthetic" and "合成" in body["data_note"]


# ---------------------------------------------------------------- startup failure

def test_app_refuses_to_start_without_artifacts(tmp_path):  # docs/04-sad.md failure handling
    with pytest.raises(RuntimeError, match="run_experiment.py"):
        with TestClient(create_app(tmp_path)):
            pass
