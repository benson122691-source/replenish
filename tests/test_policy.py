import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replenishai.policy import order_quantity, order_up_to_level, reorder_point, z_from_service_level
from replenishai.simulate import CostParams, simulate_series


def test_z_matches_known_normal_quantiles():
    assert z_from_service_level(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert z_from_service_level(0.5) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [0.4, 0.9999, 1.2])
def test_service_level_out_of_range_is_rejected(bad):  # BR-4
    with pytest.raises(ValueError):
        z_from_service_level(bad)


def test_reorder_point_formula():
    assert reorder_point(mu=3, sigma=2, lead_time=4, z=1.5) == pytest.approx(3 * 4 + 1.5 * 2 * math.sqrt(4))


def test_order_up_to_level_includes_review_period():
    assert order_up_to_level(3, 0, 6, 1.0, review=1) == pytest.approx(3 * 7)


def test_no_order_when_stock_is_enough():  # TC-9
    assert order_quantity(inventory_position=50, target=30) == 0


def test_order_respects_moq_and_is_integer():  # TC-10, TC-11
    q = order_quantity(inventory_position=28, target=30.2, moq=12)
    assert q == 12 and isinstance(q, int)
    assert order_quantity(inventory_position=10, target=30.2, moq=5) == 21  # ceil(20.2)


def test_negative_moq_is_rejected():
    with pytest.raises(ValueError):
        order_quantity(1, 5, moq=-1)


def test_simulation_has_no_stockout_with_perfect_forecast_and_constant_demand():
    demand = np.full(28, 5.0)
    r = simulate_series(demand, forecast=demand.copy(), sigma=0.0, price=2.0, lead_time=2)
    assert r["stockout_units"] == 0 and r["fill_rate"] == 1.0


def test_simulation_conserves_demand_and_costs_are_non_negative():
    rng = np.random.default_rng(0)
    demand = rng.poisson(4, 28).astype(float)
    r = simulate_series(demand, forecast=np.full(28, 4.0), sigma=2.0, price=3.0, lead_time=7)
    assert r["total_cost"] >= 0 and r["holding_cost"] >= 0 and r["stockout_cost"] >= 0
    assert 0 <= r["fill_rate"] <= 1
    assert r["demand"] == pytest.approx(demand.sum())


def test_underforecasting_increases_stockouts():
    demand = np.full(28, 6.0)
    good = simulate_series(demand, np.full(28, 6.0), 1.0, 2.0, lead_time=3)
    bad = simulate_series(demand, np.full(28, 2.0), 1.0, 2.0, lead_time=3)
    assert bad["stockout_units"] > good["stockout_units"]


def test_higher_stockout_cost_ratio_raises_total_cost_when_there_are_stockouts():
    demand = np.full(28, 6.0)
    kw = dict(forecast=np.full(28, 2.0), sigma=1.0, price=2.0, lead_time=3)
    lo = simulate_series(demand, cost=CostParams(stockout_ratio=0.2), **kw)
    hi = simulate_series(demand, cost=CostParams(stockout_ratio=1.6), **kw)
    assert hi["total_cost"] > lo["total_cost"]
