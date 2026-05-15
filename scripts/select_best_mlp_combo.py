#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse

def choose_best(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["classifier"].str.contains("MLP")]
    pivot = df.pivot_table(
        index="num_points_per_class",
        columns=["std_label", "real_ratio"],
        values="accuracy",
    )
    max_per_row = pivot.max(axis=1)
    wins = {}
    for col in pivot.columns:
        wins[col] = (pivot[col] == max_per_row).sum()
    wins_series = pd.Series(wins).sort_values(ascending=False)
    top = wins_series.index[0]
    tied = [col for col, cnt in wins.items() if cnt == wins_series.iloc[0]]
    if len(tied) > 1:
        means = {col: pivot[col].mean() for col in tied}
        top = max(means, key=means.get)
    std_label, real_ratio = top
    return std_label, float(real_ratio), int(wins_series.iloc[0])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/eval_synthetic_metrics.csv")
    args = p.parse_args()
    std_label, real_ratio, wins = choose_best(args.csv)
    print(f"selected_std_label={std_label}")
    print(f"selected_real_ratio={real_ratio}")
    print(f"wins_count={wins}")

if __name__ == "__main__":
    main()
