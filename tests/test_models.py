import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replenishai.data import data_source, load_m5, split_train_test
from replenishai.features import add_features
from replenishai.models import predict_lgbm, train_lgbm
from replenishai.synthetic import make_synthetic_m5


@pytest.fixture(scope="module")
def parts(tmp_path_factory):
    raw = make_synthetic_m5(tmp_path_factory.mktemp("m5"), n_items_per_dept=4, n_days=400)
    df = add_features(load_m5(raw))
    train, test, _ = split_train_test(df)
    return raw, train, test


def test_synthetic_folder_is_marked_synthetic(parts):  # NFR-5
    assert data_source(parts[0]) == "synthetic"


def test_lgbm_predicts_every_test_row_non_negative(parts):  # FR-4, BR-2
    _, train, test = parts
    p = predict_lgbm(train_lgbm(train, {"n_estimators": 50}), test)
    assert len(p) == len(test) and p["pred"].notna().all() and (p["pred"] >= 0).all()


def test_lgbm_is_reproducible(parts):  # NFR-1, TC-19
    _, train, test = parts
    a = predict_lgbm(train_lgbm(train, {"n_estimators": 50}), test)["pred"].to_numpy()
    b = predict_lgbm(train_lgbm(train, {"n_estimators": 50}), test)["pred"].to_numpy()
    assert np.allclose(a, b)


def test_quantile_model_is_not_below_mean_model_on_average(parts):
    """A high-quantile forecast should sit above the mean forecast (it carries safety margin)."""
    from replenishai.models import train_lgbm_quantile
    _, train, test = parts
    mean = predict_lgbm(train_lgbm(train, {"n_estimators": 50}), test)["pred"].mean()
    q = predict_lgbm(train_lgbm_quantile(train, 0.8, {"n_estimators": 50}), test)["pred"].mean()
    assert q > mean


def test_tsb_is_non_negative_and_covers_every_test_row(parts):
    from replenishai.baselines import tsb
    _, train, test = parts
    p = tsb(train, test)
    assert len(p) == len(test) and p["pred"].notna().all() and (p["pred"] >= 0).all()


def test_paired_bootstrap_ci_brackets_point_estimate_and_detects_a_clear_saving():
    import pandas as pd
    from replenishai.simulate import paired_bootstrap_ci
    rng = np.random.default_rng(1)
    base = pd.Series(rng.uniform(5, 10, 200))
    better = base * 0.8                      # exactly 20% cheaper on every series
    r = paired_bootstrap_ci(better, base)
    assert r["pct"] == pytest.approx(-20.0) and r["lo"] <= r["pct"] <= r["hi"] and r["hi"] < 0
