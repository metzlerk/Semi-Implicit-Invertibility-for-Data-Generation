#!/usr/bin/env python3
import argparse
from pathlib import Path
import runpy

import numpy as np
import pandas as pd


mod = runpy.run_path("scripts/4f-kjm-evaluatesynthetic-data.py")
build_mlp = mod["build_mlp"]
CLASS_NAMES = mod["CLASS_NAMES"]

NON_FEATURE_COLUMNS = {"Unnamed: 0", "index", "Label", "TemperatureKelvin", "PressureBar", "temp_K"}


def load_real_df(feather_path):
    df = pd.read_feather(feather_path)
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def split_features_labels(df):
    label_size = len(CLASS_NAMES)
    X = df.iloc[:, : (df.shape[1] - label_size)].values
    y = df.iloc[:, -label_size:].values.argmax(axis=1)
    return X, y


def load_synthetic_data(spectra_path, labels_path):
    spectra = np.load(spectra_path)
    labels = np.load(labels_path)
    return spectra, labels


def build_training_set(X_real, y_real, X_synthetic, y_synthetic, num_points, ratio):
    X_train = []
    y_train = []
    for class_label in np.unique(y_real):
        real_indices = np.where(y_real == class_label)[0][: int(num_points * ratio + 0.5)]
        synthetic_indices = np.where(y_synthetic == class_label)[0][: int(num_points * (1 - ratio) + 0.5)]
        if len(real_indices) > 0:
            X_train.append(X_real[real_indices])
            y_train.append(y_real[real_indices])
        if len(synthetic_indices) > 0:
            X_train.append(X_synthetic[synthetic_indices])
            y_train.append(y_synthetic[synthetic_indices])
    if not X_train or not y_train:
        return None, None
    return np.vstack(X_train), np.hstack(y_train)


def train_and_score(X_train, y_train, X_test, y_test, random_state):
    model = build_mlp(random_state=random_state)
    model.fit(X_train, y_train)
    return float((model.predict(X_test) == y_test).mean())


def build_num_points_grid(max_points, step):
    grid = list(range(step, max_points + 1, step))
    if not grid or grid[-1] != max_points:
        grid.append(max_points)
    return grid


def run_sweep(X_real, y_real, X_synthetic, y_synthetic, X_test, y_test, *, ratio, num_points_grid, classifier_name, std_label, random_state):
    rows = []
    for num_points in num_points_grid:
        X_train, y_train = build_training_set(X_real, y_real, X_synthetic, y_synthetic, num_points, ratio)
        if X_train is None or y_train is None:
            continue
        accuracy = train_and_score(X_train, y_train, X_test, y_test, random_state)
        rows.append(
            {
                "classifier": classifier_name,
                "std_label": std_label,
                "num_points_per_class": num_points,
                "real_ratio": ratio,
                "accuracy": accuracy,
                "train_datapoints": len(X_train),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Build a full-dataset MLP heatmap CSV using 1000-point increments.")
    parser.add_argument("--train-feather", default="Data/train_data.feather")
    parser.add_argument("--test-feather", default="Data/test_data.feather")
    parser.add_argument("--synthetic-spectra", default="results/generated_spectra_std1.5.npy")
    parser.add_argument("--synthetic-labels", default="results/generated_labels_std1.5.npy")
    parser.add_argument("--out-csv", default="results/full_dataset_heatmap.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--step", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    train_df = load_real_df(args.train_feather)
    test_df = load_real_df(args.test_feather)
    X_real, y_real = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)
    X_synthetic, y_synthetic = load_synthetic_data(args.synthetic_spectra, args.synthetic_labels)

    max_points = int(np.bincount(y_real).min())
    num_points_grid = build_num_points_grid(max_points, args.step)

    rows = []
    rows.extend(
        run_sweep(
            X_real,
            y_real,
            X_synthetic,
            y_synthetic,
            X_test,
            y_test,
            ratio=0.6,
            num_points_grid=num_points_grid,
            classifier_name="MLP_selected",
            std_label="std1.5",
            random_state=args.random_state,
        )
    )
    rows.extend(
        run_sweep(
            X_real,
            y_real,
            X_synthetic,
            y_synthetic,
            X_test,
            y_test,
            ratio=1.0,
            num_points_grid=num_points_grid,
            classifier_name="MLP_selected",
            std_label="std1.5",
            random_state=args.random_state,
        )
    )
    rows.extend(
        run_sweep(
            X_real,
            y_real,
            X_real,
            y_real,
            X_test,
            y_test,
            ratio=1.0,
            num_points_grid=num_points_grid,
            classifier_name="MLP_realonly",
            std_label="realonly",
            random_state=args.random_state,
        )
    )

    out_df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"saved {len(out_df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
