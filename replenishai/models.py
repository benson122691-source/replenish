"""LightGBM demand model (FR-4). One global model for all series (as in M5 top solutions)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL, FEATURES

PARAMS = dict(
    objective="poisson",          # sales are non-negative counts
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, deterministic=True, force_row_wise=True,
    n_jobs=4, verbose=-1,
)


def train_lgbm(train: pd.DataFrame, params: dict | None = None) -> lgb.LGBMRegressor:
    """Fit on rows whose lag_28 exists (the first 28 days of each series have no history)."""
    rows = train.dropna(subset=["lag_28"])
    model = lgb.LGBMRegressor(**{**PARAMS, **(params or {})})
    model.fit(rows[FEATURES], rows["sales"], categorical_feature=CATEGORICAL)
    return model


def predict_lgbm(model: lgb.LGBMRegressor, test: pd.DataFrame) -> pd.DataFrame:
    """Forecast for every test row; negative values are clipped to 0 (BR-2)."""
    pred = np.clip(model.predict(test[FEATURES]), 0, None)
    return pd.DataFrame({"id": test["id"].values, "date": test["date"].values, "pred": pred})


def train_lgbm_quantile(train: pd.DataFrame, alpha: float, params: dict | None = None) -> lgb.LGBMRegressor:
    """Daily-demand quantile model. Used as a risk-aware order-up-to input instead of mean + z*sigma
    (which assumes normal demand; real retail demand is skewed and has many zeros)."""
    return train_lgbm(train, {"objective": "quantile", "alpha": alpha, **(params or {})})
