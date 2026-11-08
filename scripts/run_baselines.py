#!/usr/bin/env python3
"""M1 entry point: build (or reuse) data, add features, score the baselines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replenishai.baselines import moving_average, score, seasonal_naive
from replenishai.data import load_m5, split_train_test
from replenishai.features import add_features
from replenishai.synthetic import make_synthetic_m5

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    if not (RAW / "sales_train_evaluation.csv").exists():
        print("No M5 files found -> generating SYNTHETIC M5-format data in data/raw/")
        make_synthetic_m5(RAW)
    df = load_m5(RAW)
    print(f"series: {df['id'].nunique()}, rows: {len(df):,}, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")
    df = add_features(df)
    train, test, origin = split_train_test(df)
    print(f"forecast origin: {origin.date()}, hold-out days: {test['date'].nunique()}\n")

    for name, fn in (("seasonal_naive", seasonal_naive), ("moving_avg_28", moving_average)):
        print(f"{name:<16}", {k: round(v, 4) for k, v in score(test, fn(train, test), train).items()})


if __name__ == "__main__":
    main()
