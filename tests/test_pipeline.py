import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replenishai.baselines import moving_average, score, seasonal_naive
from replenishai.data import HORIZON, load_m5, split_train_test
from replenishai.features import FEATURES, add_features, assert_no_future_leak
from replenishai.synthetic import make_synthetic_m5


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return make_synthetic_m5(tmp_path_factory.mktemp("m5"), n_items_per_dept=4, n_days=400)


@pytest.fixture(scope="module")
def df(raw):
    return add_features(load_m5(raw))


def test_synthetic_files_follow_m5_schema(raw):
    wide = pd.read_csv(raw / "sales_train_evaluation.csv")
    assert {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d_1", "d_400"} <= set(wide.columns)
    assert {"d", "date", "wm_yr_wk", "wday", "event_name_1"} <= set(pd.read_csv(raw / "calendar.csv").columns)
    assert {"store_id", "item_id", "wm_yr_wk", "sell_price"} <= set(pd.read_csv(raw / "sell_prices.csv").columns)


def test_load_is_tidy(raw):
    long = load_m5(raw)
    assert long["sales"].ge(0).all()
    assert not long.duplicated(["id", "date"]).any()
    assert long["sell_price"].notna().all()


def test_features_have_no_future_leak(df):
    assert_no_future_leak(df)
    assert set(FEATURES) <= set(df.columns)


def test_split_holds_out_exactly_horizon_days(df):
    train, test, origin = split_train_test(df)
    assert test["date"].nunique() == HORIZON
    assert train["date"].max() == origin
    assert train["date"].max() < test["date"].min()


def test_perfect_forecast_scores_zero(df):
    train, test, _ = split_train_test(df)
    perfect = test[["id", "date"]].assign(pred=test["sales"].values)
    s = score(test, perfect, train)
    assert s["rmse"] == pytest.approx(0) and s["rmsse"] == pytest.approx(0)


def test_baselines_cover_every_test_row(df):
    train, test, _ = split_train_test(df)
    for fn in (seasonal_naive, moving_average):
        p = fn(train, test)
        assert len(p) == len(test) and p["pred"].notna().all()


def test_seasonal_naive_is_exact_on_a_noise_free_weekly_series():
    """Implementation check on deterministic data (real data is noisy, so a single-week
    naive forecast can lose to a 28-day average - that is a property of the data, not a bug)."""
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    pattern = np.array([1, 2, 3, 4, 5, 10, 12], dtype=float)
    d = pd.DataFrame({"id": "A", "date": dates,
                      "sales": np.tile(pattern, 10)})
    train, test = d[d["date"] <= "2024-02-11"], d[d["date"] > "2024-02-11"]
    p = seasonal_naive(train, test)
    assert np.allclose(p["pred"].to_numpy(), test["sales"].to_numpy())
    assert score(test, moving_average(train, test), train)["rmse"] > 0
