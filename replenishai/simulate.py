"""Inventory simulation (FR-6): replay real demand against a forecast-driven policy.

Rules (kept simple on purpose, all stated in docs/05-sds.md):
- Review every day at the END of the day. If inventory position (on hand + on order) is below
  the order-up-to level S, order the gap (at least MOQ). The order arrives L days later.
- Demand beyond on-hand stock is LOST (lost sales), not backordered.
- Cost = holding (per unit per day, on end-of-day stock) + stockout (per lost unit).
- All strategies share the same starting stock, sigma, service level and cost parameters, so the
  ONLY difference between strategies is the forecast. That isolates "does a better forecast save money".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .policy import order_quantity, order_up_to_level, z_from_service_level


@dataclass(frozen=True)
class CostParams:
    """ASSUMED values (待確認): there is no real cost data in M5.
    holding: 25% of unit price per year; stockout: `stockout_ratio` x price (lost margin)."""
    holding_rate_per_year: float = 0.25
    stockout_ratio: float = 0.4

    def holding_per_unit_day(self, price: float) -> float:
        return self.holding_rate_per_year / 365 * price

    def stockout_per_unit(self, price: float) -> float:
        return self.stockout_ratio * price


def simulate_series(
    demand: np.ndarray, forecast: np.ndarray, sigma: float, price: float,
    lead_time: int = 7, service_level: float = 0.95, moq: int = 0,
    start_stock: int | None = None, cost: CostParams = CostParams(), review: int = 1,
) -> dict:
    z = z_from_service_level(service_level)
    T = len(demand)
    on_hand = int(math.ceil(np.mean(forecast[:max(lead_time, 1)]) * lead_time)) if start_stock is None else start_stock
    arrivals = np.zeros(T + lead_time + 2)
    h, p = cost.holding_per_unit_day(price), cost.stockout_per_unit(price)
    lost = held = 0.0

    for t in range(T):
        on_hand += int(arrivals[t])
        sold = min(on_hand, float(demand[t]))
        lost += float(demand[t]) - sold
        on_hand -= sold
        held += h * on_hand

        window = forecast[t + 1: t + 1 + lead_time + review]   # days this order must cover
        if len(window) == 0:
            continue
        S = order_up_to_level(float(np.mean(window)), sigma, lead_time, z, review)
        position = on_hand + arrivals[t + 1: t + lead_time + 1].sum()
        q = order_quantity(position, S, moq)
        if q > 0:
            arrivals[t + lead_time] += q

    total_demand = float(np.sum(demand))
    return {
        "stockout_units": lost, "holding_cost": held, "stockout_cost": lost * p,
        "total_cost": held + lost * p,
        "fill_rate": 1 - lost / total_demand if total_demand > 0 else 1.0,
        "demand": total_demand,
    }


def series_stats(train: pd.DataFrame, window: int = 56) -> pd.DataFrame:
    """Per-series sigma (std of the last `window` training days) and last known price.
    Shared by the simulation and the API so both use exactly the same sigma."""
    g = train.sort_values("date").groupby("id")
    return pd.DataFrame({
        "sigma": g["sales"].apply(lambda s: s.tail(window).std(ddof=0)),
        "price": g["sell_price"].last(),
    })


def simulate_strategy(
    test: pd.DataFrame, pred: pd.DataFrame, train: pd.DataFrame, **kw,
) -> dict:
    """Run every series in the hold-out for one forecast; return totals across series."""
    stats = series_stats(train)
    sigma, price = stats["sigma"], stats["price"]
    m = test[["id", "date", "sales"]].merge(pred, on=["id", "date"]).sort_values(["id", "date"])
    rows = []
    for sid, g in m.groupby("id"):
        rows.append(simulate_series(g["sales"].to_numpy(), g["pred"].to_numpy(),
                                    float(sigma.get(sid, 0.0)), float(price[sid]), **kw))
    df = pd.DataFrame(rows)
    tot = df[["stockout_units", "holding_cost", "stockout_cost", "total_cost", "demand"]].sum()
    return {**tot.to_dict(), "fill_rate": float(1 - tot["stockout_units"] / tot["demand"])}
