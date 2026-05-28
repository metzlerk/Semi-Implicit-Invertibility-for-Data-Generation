#!/usr/bin/env python3
"""Prepare a split file into the expected 1676-feature layout and run the encoder latent builder.

Usage:
  python scripts/prepare_and_run_latent_extraction.py --split Data/splits/split_1_of_5.feather
"""
import argparse
import os
import re
import subprocess
import pandas as pd


def find_feature_cols(df):
    p_cols = [c for c in df.columns if re.match(r"^p_\d+", c)]
    n_cols = [c for c in df.columns if re.match(r"^n_\d+", c)]
    def sort_key(name):
        return int(name.split("_")[1])
    p_cols = sorted(p_cols, key=sort_key)
    n_cols = sorted(n_cols, key=sort_key)
    return p_cols + n_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--test", default="Data/test_data_with_conditions.feather")
    parser.add_argument("--out-prefix", default="results/autoencoder_train_latent_split")
    parser.add_argument("--split-index", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_feather(args.split)
    feat = find_feature_cols(df)
    if len(feat) != 1676:
        raise SystemExit(f"Found {len(feat)} feature columns, expected 1676. Columns found: {len(feat)}")

    # ensure index and Label are present
    if "index" not in df.columns:
        df.insert(0, "index", range(len(df)))
    if "Label" not in df.columns:
        raise SystemExit("Label column not found in split; cannot proceed.")

    clean_df = df[["index"] + feat + ["Label"]].copy()
    split_basename = os.path.basename(args.split).replace('.feather','')
    clean_path = os.path.join('Data', f'clean_{split_basename}.feather')
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    clean_df.to_feather(clean_path)
    print(f"Wrote cleaned split to {clean_path}")

    # also ensure test data cleaned
    clean_test = os.path.join('Data', 'clean_test_data.feather')
    if not os.path.exists(clean_test):
        test_df = pd.read_feather(args.test)
        test_feat = find_feature_cols(test_df)
        if len(test_feat) != 1676:
            raise SystemExit(f"Test data has {len(test_feat)} features, expected 1676")
        if "index" not in test_df.columns:
            test_df.insert(0, "index", range(len(test_df)))
        if "Label" not in test_df.columns:
            raise SystemExit("Label missing in test data")
        test_df = test_df[["index"] + test_feat + ["Label"]].copy()
        test_df.to_feather(clean_test)
        print(f"Wrote cleaned test data to {clean_test}")

    # run encoder builder
    idx = args.split_index if args.split_index is not None else split_basename
    out_train = f"results/autoencoder_train_latent_{idx}.npy"
    out_test = f"results/autoencoder_test_latent_{idx}.npy"
    cmd = [
        "python3",
        "scripts/build_autoencoder_latents.py",
        "--train-data",
        clean_path,
        "--test-data",
        clean_test,
        "--output-train",
        out_train,
        "--output-test",
        out_test,
        "--save-csv",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == '__main__':
    main()
