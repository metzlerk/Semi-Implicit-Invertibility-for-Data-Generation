#!/usr/bin/env python3
"""Create 5 temperature-based splits (20% each) from a feather dataset.

Usage: python scripts/create_temp_splits.py --input Data/train_data_with_conditions.feather --out_dir Data/splits
"""
import argparse
import os
import sys
import pandas as pd


def find_temp_column(df):
    candidates = ["temperature", "temp", "T", "Temperature", "temp_C"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try any column containing 'temp' or 'temperature'
    for c in df.columns:
        if "temp" in c.lower() or "temperature" in c.lower():
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--column", default=None)
    args = p.parse_args()

    inp = args.input
    out_dir = args.out_dir
    n = args.n_splits

    os.makedirs(out_dir, exist_ok=True)

    print(f"Reading {inp}")
    df = pd.read_feather(inp)
    print(f"Rows: {len(df)}, columns: {list(df.columns)}")

    col = args.column or find_temp_column(df)
    if col is None:
        print("ERROR: could not find a temperature column. Available columns:")
        for c in df.columns:
            print(" -", c)
        sys.exit(2)

    print(f"Using temperature column: {col}")

    quantiles = [i / n for i in range(n + 1)]
    bounds = df[col].quantile(quantiles).values
    print("Quantile bounds:")
    for i, q in enumerate(quantiles):
        print(f"  {q:.2f}: {bounds[i]:.6f}")

    for i in range(n):
        lo = bounds[i]
        hi = bounds[i + 1]
        if i < n - 1:
            mask = (df[col] >= lo) & (df[col] < hi)
        else:
            mask = (df[col] >= lo) & (df[col] <= hi)
        split_df = df[mask].copy()
        out_path = os.path.join(out_dir, f"split_{i+1}_of_{n}.feather")
        split_df.reset_index(drop=True, inplace=True)
        split_df.to_feather(out_path)
        print(f"Wrote split {i+1}: {len(split_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
