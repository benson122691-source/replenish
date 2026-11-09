#!/usr/bin/env python3
"""M2 experiment: forecast accuracy AND replenishment cost for 4 strategies, over several
rolling origins (each origin = retrain + 28-day hold-out), plus a cost-ratio sensitivity table.

Strategies: seasonal_naive, moving_avg_28, lightgbm, oracle (perfect forecast = lower bound).
Outputs: artifacts/results.json, artifacts/holdout_forecasts.csv, docs/results.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from replenishai.baselines import moving_average, score, seasonal_naive, tsb
from replenishai.data import HORIZON, data_source, load_m5
from replenishai.features import add_features
from replenishai.models import predict_lgbm, train_lgbm, train_lgbm_quantile
from replenishai.simulate import (CostParams, paired_bootstrap_ci, series_stats,
                                  simulate_per_series, simulate_strategy)
from replenishai.synthetic import make_synthetic_m5

QUANTILE = 0.8
RAW = ROOT / "data" / "raw"
STRATEGIES = ["seasonal_naive", "moving_avg_28", "tsb", "lightgbm", "lightgbm_q", "oracle"]
# lightgbm_q: daily-demand quantile forecast as the order-up-to input, sigma term switched off
SIGMA_SCALE = {"lightgbm_q": 0.0}
REAL = [s for s in STRATEGIES if s != "oracle"]
STOCKOUT_RATIOS = [0.2, 0.4, 0.8, 1.6]


def forecasts_for_origin(df: pd.DataFrame, origin: pd.Timestamp):
    train = df[df["date"] <= origin]
    test = df[(df["date"] > origin) & (df["date"] <= origin + pd.Timedelta(days=HORIZON))]
    preds = {
        "seasonal_naive": seasonal_naive(train, test),
        "moving_avg_28": moving_average(train, test),
        "tsb": tsb(train, test),
        "lightgbm": predict_lgbm(train_lgbm(train), test),
        "lightgbm_q": predict_lgbm(train_lgbm_quantile(train, QUANTILE), test),
        "oracle": test[["id", "date"]].assign(pred=test["sales"].values),
    }
    return train, test, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=3, help="number of rolling 28-day origins")
    ap.add_argument("--lead-time", type=int, default=7)
    ap.add_argument("--service-level", type=float, default=0.95)
    ap.add_argument("--quantile", type=float, default=0.8, help="alpha of the lightgbm_q daily model")
    ap.add_argument("--stores", nargs="*", default=None, help="subset of M5 stores, e.g. CA_1")
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    if not (RAW / "sales_train_evaluation.csv").exists():
        print("No M5 files found -> generating SYNTHETIC M5-format data in data/raw/")
        make_synthetic_m5(RAW)
    source = data_source(RAW)
    global QUANTILE
    QUANTILE = args.quantile
    df = add_features(load_m5(RAW, stores=args.stores, max_items=args.max_items))
    last = df["date"].max()
    origins = [last - pd.Timedelta(days=HORIZON * k) for k in range(args.origins, 0, -1)]
    print(f"data source: {source.upper()} | series: {df['id'].nunique()} | origins: "
          f"{[str(o.date()) for o in origins]}\n")

    acc_rows, cost_rows, sens_rows, per_series = [], [], [], []
    for i, origin in enumerate(origins):
        train, test, preds = forecasts_for_origin(df, origin)
        for name, p in preds.items():
            acc_rows.append({"origin": str(origin.date()), "strategy": name, **score(test, p, train)})
            sc = SIGMA_SCALE.get(name, 1.0)
            ps = simulate_per_series(test, p, train, sigma_scale=sc, lead_time=args.lead_time,
                                     service_level=args.service_level, cost=CostParams())
            per_series.append(ps.assign(origin=str(origin.date()), strategy=name))
            c = ps[["stockout_units", "holding_cost", "stockout_cost", "total_cost", "demand"]].sum().to_dict()
            c["fill_rate"] = float(1 - c["stockout_units"] / c["demand"])
            cost_rows.append({"origin": str(origin.date()), "strategy": name, **c})
            for ratio in STOCKOUT_RATIOS:
                s = simulate_strategy(test, p, train, sigma_scale=sc, lead_time=args.lead_time,
                                      service_level=args.service_level,
                                      cost=CostParams(stockout_ratio=ratio))
                sens_rows.append({"origin": str(origin.date()), "strategy": name,
                                  "stockout_ratio": ratio, "total_cost": s["total_cost"]})
        if i == len(origins) - 1:   # keep the latest window for the API (M3)
            out = test[["id", "item_id", "store_id", "date", "sales"]].copy()
            for name, p in preds.items():
                out = out.merge(p.rename(columns={"pred": f"pred_{name}"}), on=["id", "date"])
            (ROOT / "artifacts").mkdir(exist_ok=True)
            out.to_csv(ROOT / "artifacts" / "holdout_forecasts.csv", index=False)
            # sigma/price per series at this origin: the API's /replenish needs them (M3)
            series_stats(train).rename_axis("id").reset_index().to_csv(
                ROOT / "artifacts" / "series_stats.csv", index=False)
        print(f"origin {origin.date()} done")

    acc = pd.DataFrame(acc_rows).groupby("strategy")[["rmse", "mae", "bias", "rmsse"]].mean().loc[STRATEGIES]
    cost = pd.DataFrame(cost_rows).groupby("strategy")[
        ["total_cost", "holding_cost", "stockout_cost", "stockout_units", "fill_rate"]].mean().loc[STRATEGIES]
    base = cost.loc["moving_avg_28", "total_cost"]
    cost["vs_ma28_pct"] = (cost["total_cost"] / base - 1) * 100
    sens = (pd.DataFrame(sens_rows).groupby(["stockout_ratio", "strategy"])["total_cost"].mean()
              .unstack("strategy")[STRATEGIES])
    sens["best_non_oracle"] = sens[REAL].idxmin(axis=1)

    ps_all = pd.concat(per_series, ignore_index=True)
    tot = ps_all.groupby(["strategy", "id"])["total_cost"].sum().unstack("strategy")
    boot = {k: paired_bootstrap_ci(tot[k], tot["moving_avg_28"]) for k in REAL if k != "moving_avg_28"}

    print("\nForecast accuracy (mean over origins; lower is better)\n", acc.round(3))
    print("\nReplenishment cost (mean per 28-day window, summed over series)\n", cost.round(3))
    print("\nSensitivity: total cost by stockout cost ratio\n", sens.round(1))
    print("\nPaired bootstrap vs moving_avg_28 (% cost change, 95% CI over series)")
    for k, v in boot.items():
        print(f"  {k:15s} {v['pct']:+.1f}%  [{v['lo']:+.1f}, {v['hi']:+.1f}]")

    results = {"data_source": source, "origins": [str(o.date()) for o in origins],
               "lead_time": args.lead_time, "service_level": args.service_level,
               "cost_params": asdict(CostParams()),   # ASSUMED values, see simulate.CostParams
               "quantile": args.quantile, "bootstrap_vs_ma28": boot,
               "accuracy": acc.round(4).to_dict("index"), "cost": cost.round(3).to_dict("index"),
               "sensitivity": sens.round(2).reset_index().to_dict("records")}
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_results_md(results, acc, cost, sens, boot)


def write_results_md(results: dict, acc, cost, sens, boot) -> None:
    label = ("**資料來源：合成資料（synthetic）。以下數字不代表真實零售表現，不可當作真實成果引用。**"
             if results["data_source"] == "synthetic" else "資料來源：M5（Walmart）。")
    md = [
        "# 實驗結果（自動產生）", "", label, "",
        f"- 滾動原點：{', '.join(results['origins'])}（各 28 天保留期，每個原點重新訓練）",
        f"- 交期 {results['lead_time']} 天；服務水準 {results['service_level']}；每日盤點",
        "- 成本參數為**假設值（待確認）**：持有成本＝年化 25% × 單價；缺貨成本＝比例 × 單價（預設 0.4）",
        "- 所有策略共用相同的起始庫存、σ、服務水準與成本；差別只在預測。oracle＝完美預測（成本下界）", "",
        "## 預測誤差（越低越好）", "", acc.round(3).to_markdown(), "",
        "## 補貨成本（每個 28 天窗口，所有序列加總；平均於各原點）", "", cost.round(3).to_markdown(), "",
        "## 敏感度：缺貨成本比例改變時的總成本", "", sens.round(1).to_markdown(), "",
        "## 與 moving_avg_28 的總成本差異（以序列為單位的配對 bootstrap，95% 信賴區間）", "",
        "| strategy | 成本變化 % | 下界 | 上界 | 區間是否不含 0 |", "|:--|--:|--:|--:|:--|",
        *[f"| {k} | {v['pct']:+.1f} | {v['lo']:+.1f} | {v['hi']:+.1f} | "
          f"{'是' if (v['lo'] > 0 or v['hi'] < 0) else '否（不顯著）'} |" for k, v in boot.items()], "",
        f"- lightgbm_q：每日需求的 {results['quantile']:.2f} 分位數預測直接當補貨輸入，不加 zσ。", "",
    ]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
