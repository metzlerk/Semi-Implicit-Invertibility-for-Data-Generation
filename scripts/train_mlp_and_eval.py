#!/usr/bin/env python3
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import runpy

# reuse build_mlp and CLASS_NAMES from existing evaluator script
mod = runpy.run_path("scripts/4f-kjm-evaluatesynthetic-data.py")
build_mlp = mod["build_mlp"]
CLASS_NAMES = mod["CLASS_NAMES"]

def load_real_df(feather_path):
    df = pd.read_feather(feather_path)
    df = df.drop(columns=[c for c in ["Unnamed: 0", "index", "Label"] if c in df.columns], errors="ignore")
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
    p.add_argument("--std-label", default="selected")
    p.add_argument("--ratio", type=float, default=1.0)
    p.add_argument("--save-model", default=None)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    real_df = load_real_df(args.real_feather)
    test_df = load_real_df(args.test_feather)
    X_real, y_real = df_features_labels_from_real(real_df)
    X_test, y_test = df_features_labels_from_real(test_df)

    if args.synthetic_spectra and args.synthetic_labels:
        X_syn, y_syn = load_synthetic_np(args.synthetic_spectra, args.synthetic_labels)
    else:
        X_syn = np.empty((0, X_real.shape[1]))
        y_syn = np.empty((0,), dtype=int)

    X_train, y_train = build_combined_training(X_real, y_real, X_syn, y_syn, args.ratio)

    model = build_mlp(random_state=args.random_state)
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
