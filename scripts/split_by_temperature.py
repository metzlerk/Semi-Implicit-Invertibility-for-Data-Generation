#!/usr/bin/env python3
import argparse
import pandas as pd

def split(in_file, out_train, out_test, temp_col="temp_K", train_quantile=0.8):
    df = pd.read_feather(in_file)
    if temp_col not in df.columns:
        candidate_cols = [c for c in df.columns if "temp" in c.lower() or "kelvin" in c.lower()]
        if len(candidate_cols) == 1:
            temp_col = candidate_cols[0]
        elif len(candidate_cols) > 1:
            temp_col = candidate_cols[0]
        else:
            raise KeyError(f"{temp_col} not in {in_file} columns and no temperature-like column found")
    threshold = df[temp_col].quantile(train_quantile)
    train_df = df[df[temp_col] <= threshold].reset_index(drop=True)
    test_df = df[df[temp_col] > threshold].reset_index(drop=True)
    train_df.to_feather(out_train)
    test_df.to_feather(out_test)
    print(f"wrote {len(train_df)} train / {len(test_df)} test rows (threshold={threshold})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", required=True)
    p.add_argument("--out-train", required=True)
    p.add_argument("--out-test", required=True)
    p.add_argument("--temp-col", default="temp_K")
    p.add_argument("--train-quantile", type=float, default=0.8)
    args = p.parse_args()
    split(args.in_file, args.out_train, args.out_test, args.temp_col, args.train_quantile)
