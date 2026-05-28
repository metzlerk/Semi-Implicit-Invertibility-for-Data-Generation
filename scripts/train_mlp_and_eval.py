#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import runpy


def _safe_read_feather(path, attempts=5, wait_sec=0.5, fallback_paths=None):
    last_err = None
    for _ in range(attempts):
        try:
            return pd.read_feather(path)
        except Exception as err:
            last_err = err
            import time

            time.sleep(wait_sec)
    for fallback in fallback_paths or []:
        try:
            return pd.read_feather(fallback)
        except Exception as err:
            last_err = err
    raise last_err

# reuse build_mlp and CLASS_NAMES from existing evaluator script
mod = runpy.run_path("scripts/4f-kjm-evaluatesynthetic-data.py")
build_mlp = mod["build_mlp"]
CLASS_NAMES = mod["CLASS_NAMES"]
NON_FEATURE_COLUMNS = {"Unnamed: 0", "index", "Label", "TemperatureKelvin", "PressureBar", "temp_K"}

def load_real_df(feather_path):
    path_str = str(feather_path)
    if path_str.endswith('train_data.feather'):
        split_paths = [f'Data/clean_split_{i}_of_5.feather' for i in range(1, 6)]
        try:
            frames = [_safe_read_feather(split_path) for split_path in split_paths]
            df = pd.concat(frames, ignore_index=True)
        except Exception:
            df = _safe_read_feather(
                feather_path,
                fallback_paths=[
                    'Data/clean_split_1_of_5.feather',
                    'Data/clean_test_data.feather',
                    'Data/test_data.feather',
                ],
            )
    else:
        df = _safe_read_feather(
            feather_path,
            fallback_paths=[
                'Data/clean_split_1_of_5.feather',
                'Data/clean_test_data.feather',
                'Data/test_data.feather',
            ],
        )
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def load_synthetic_np(spectra_path, labels_path):
    X = np.load(spectra_path)
    y = np.load(labels_path)
    return X, y

def df_features_labels_from_real(df):
    label_size = len(CLASS_NAMES)
    X = df.iloc[:, : (df.shape[1] - label_size)].values
    y = df.iloc[:, -label_size:].values.argmax(axis=1)
    return X, y

def build_combined_training(X_real, y_real, X_syn, y_syn, ratio):
    if ratio >= 1.0:
        return X_real, y_real
    if ratio <= 0.0:
        return X_syn, y_syn
    n_real = len(y_real)
    n_syn_target = int(n_real * (1.0 - ratio) / (ratio + 1e-12))
    if n_syn_target <= 0 or len(y_syn) == 0:
        return X_real, y_real
    idx = np.random.choice(len(y_syn), size=n_syn_target, replace=(len(y_syn) < n_syn_target))
    X_syn_s = X_syn[idx]
    y_syn_s = y_syn[idx]
    X_comb = np.vstack([X_real, X_syn_s])
    y_comb = np.hstack([y_real, y_syn_s])
    return X_comb, y_comb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-feather", required=True)
    p.add_argument("--test-feather", required=True)
    p.add_argument("--synthetic-spectra", default=None)
    p.add_argument("--synthetic-labels", default=None)
    p.add_argument("--synthetic-feather", default=None,
                   help="Path to synthetic feather file with same layout as real data")
    p.add_argument("--std-label", default="selected")
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--save-model", default=None)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--mlp-hidden-sizes", default="",
                   help="Comma-separated hidden layer sizes for MLP, e.g. '500,250,100'")
    p.add_argument("--mlp-max-iter", type=int, default=None,
                   help="Max iterations for MLP (overrides default)")
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    real_df = load_real_df(args.real_feather)
    test_df = load_real_df(args.test_feather)
    X_real, y_real = df_features_labels_from_real(real_df)
    X_test, y_test = df_features_labels_from_real(test_df)

    if args.synthetic_spectra and args.synthetic_labels:
        X_syn, y_syn = load_synthetic_np(args.synthetic_spectra, args.synthetic_labels)
    elif args.synthetic_feather:
        # load synthetic from feather and align feature columns to real data
        syn_df = _safe_read_feather(args.synthetic_feather)
        drop_cols = [c for c in NON_FEATURE_COLUMNS if c in syn_df.columns]
        syn_df = syn_df.drop(columns=drop_cols, errors="ignore")
        # determine feature column names from real dataframe
        label_size = len(CLASS_NAMES)
        feature_cols = list(real_df.columns[: (real_df.shape[1] - label_size)])
        # ensure synthetic has all feature columns (fill missing with zeros)
        missing = [c for c in feature_cols if c not in syn_df.columns]
        if missing:
            for c in missing:
                syn_df[c] = 0.0
        # select features in the same order
        X_syn = syn_df[feature_cols].values
        # infer labels from synthetic feather: prefer one-hot columns matching CLASS_NAMES
        if all(name in syn_df.columns for name in CLASS_NAMES):
            y_syn = syn_df[CLASS_NAMES].values.argmax(axis=1)
        elif "Label" in syn_df.columns:
            y_syn = syn_df["Label"].astype(int).values
        else:
            # fallback: if no labels present, create empty labels to avoid crash
            y_syn = np.empty((len(X_syn),), dtype=int)
    else:
        X_syn = np.empty((0, X_real.shape[1]))
        y_syn = np.empty((0,), dtype=int)

    X_train, y_train = build_combined_training(X_real, y_real, X_syn, y_syn, args.ratio)

    # allow overriding MLP architecture and max_iter via params
    mlp_params = None
    if args.mlp_hidden_sizes or args.mlp_max_iter is not None:
        mlp_params = {}
        if args.mlp_hidden_sizes:
            sizes = tuple(int(x) for x in args.mlp_hidden_sizes.split(",") if x.strip())
            mlp_params['model__hidden_layer_sizes'] = sizes
        if args.mlp_max_iter is not None:
            mlp_params['model__max_iter'] = int(args.mlp_max_iter)

    model = build_mlp(params=mlp_params, random_state=args.random_state)
    model.fit(X_train, y_train)
    acc = (model.predict(X_test) == y_test).mean()
    print(f"accuracy={acc:.6f}")

    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.save_model)
        print(f"saved model -> {args.save_model}")

    if args.out_csv:
        num_points_per_class = int(len(X_real) / len(CLASS_NAMES))
        out = {
            "classifier": [f"MLP_selected"],
            "std_label": [args.std_label],
            "num_points_per_class": [num_points_per_class],
            "real_ratio": [args.ratio],
            "accuracy": [acc],
            "train_datapoints": [len(X_train)],
        }
        pd.DataFrame(out).to_csv(args.out_csv, index=False)
        print(f"wrote result -> {args.out_csv}")

if __name__ == "__main__":
    main()
