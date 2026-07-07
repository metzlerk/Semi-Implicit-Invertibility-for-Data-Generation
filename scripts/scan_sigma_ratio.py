#!/usr/bin/env python3
"""Grid search over synthetic noise std and real/synthetic ratio for classifier accuracy.

Writes results to a CSV at `--out-csv`.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import runpy


def _safe_read_feather(path, attempts=3, wait_sec=0.2, fallback_paths=None):
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


def load_real_df(feather_path, max_rows=None, random_state=42):
    path_str = str(feather_path)
    if path_str.endswith('train_data.feather'):
        # Prefer a single clean split for speed; sampling across all splits is much slower.
        candidate_paths = [
            'Data/clean_split_5_of_5.feather',
            'Data/clean_split_4_of_5.feather',
            'Data/clean_split_3_of_5.feather',
            'Data/clean_split_2_of_5.feather',
            'Data/clean_split_1_of_5.feather',
        ]
        df = None
        for candidate in candidate_paths:
            try:
                df = _safe_read_feather(candidate)
                break
            except Exception:
                continue
        if df is None:
            df = _safe_read_feather(
                feather_path,
                fallback_paths=[
                    'Data/clean_split_1_of_5.feather',
                    'Data/clean_test_data.feather',
                    'Data/test_data.feather',
                ],
            )
        if max_rows is not None and max_rows > 0 and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=random_state)
    else:
        df = _safe_read_feather(
            feather_path,
            fallback_paths=[
                'Data/clean_split_1_of_5.feather',
                'Data/clean_test_data.feather',
                'Data/test_data.feather',
            ],
        )
    return df


def df_features_labels_from_real(df, CLASS_NAMES):
    feature_cols = [c for c in df.columns if c not in {"Label", "index", "__index_level_0__"}]
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_feature_cols].to_numpy(dtype=np.float32, copy=False)
    if "Label" in df.columns:
        y = pd.to_numeric(df["Label"], errors='coerce').fillna(0).astype(int).to_numpy()
    else:
        y_df = df[[c for c in CLASS_NAMES if c in df.columns]]
        y = y_df.to_numpy(dtype=np.float32, copy=False).argmax(axis=1)
    return X, y


def load_synthetic_from_feather(path, real_df, CLASS_NAMES):
    syn_df = _safe_read_feather(path)
    # drop known non-feature cols if present
    for c in ["Unnamed: 0", "index", "Label", "TemperatureKelvin", "PressureBar", "temp_K", "__index_level_0__"]:
        if c in syn_df.columns:
            syn_df = syn_df.drop(columns=[c], errors='ignore')
    feature_cols = [c for c in real_df.columns if c not in {"Label", "index", "__index_level_0__"}]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(real_df[c])]
    missing = [c for c in feature_cols if c not in syn_df.columns]
    if missing:
        for c in missing:
            syn_df[c] = 0.0
    X_syn = syn_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    if all(name in syn_df.columns for name in CLASS_NAMES):
        y_syn = syn_df[CLASS_NAMES].to_numpy(dtype=np.float32, copy=False).argmax(axis=1)
    elif 'Label' in syn_df.columns:
        y_syn = syn_df['Label'].astype(int).values
    else:
        y_syn = np.empty((len(X_syn),), dtype=int)
    return X_syn, y_syn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--synthetic-feather', required=True)
    p.add_argument('--real-feather', default='Data/train_data.feather')
    p.add_argument('--test-feather', default='Data/test_data.feather')
    p.add_argument('--out-csv', default='results/sigma_ratio_grid_split5.csv')
    p.add_argument('--sigmas', default='0.5,1.0,1.5')
    p.add_argument('--ratios', default='1.0,0.6,0.4,0.2')
    p.add_argument('--max-train', type=int, default=100000,
                   help='Max number of real examples to use (subsample)')
    p.add_argument('--mlp-max-iter', type=int, default=200,
                   help='Max iterations for MLP training to control runtime')
    p.add_argument('--max-test', type=int, default=20000,
                   help='Max number of test examples to read (subsample)')
    p.add_argument('--random-state', type=int, default=42)
    args = p.parse_args()

    mod = runpy.run_path('scripts/4f-kjm-evaluatesynthetic-data.py')
    build_mlp = mod['build_mlp']
    CLASS_NAMES = mod['CLASS_NAMES']

    real_df = load_real_df(args.real_feather, max_rows=args.max_train, random_state=args.random_state)
    test_df = load_real_df(args.test_feather, max_rows=args.max_test, random_state=args.random_state)
    X_real_full, y_real_full = df_features_labels_from_real(real_df, CLASS_NAMES)
    X_test, y_test = df_features_labels_from_real(test_df, CLASS_NAMES)

    X_syn_base, y_syn_base = load_synthetic_from_feather(args.synthetic_feather, real_df, CLASS_NAMES)

    sigmas = [float(x) for x in args.sigmas.split(',') if x.strip()]
    ratios = [float(x) for x in args.ratios.split(',') if x.strip()]

    rng = np.random.RandomState(args.random_state)

    results = []

    # subsample real training set for speed
    n_real_full = len(y_real_full)
    n_real = min(n_real_full, args.max_train)
    if n_real < n_real_full:
        idx_real = rng.choice(n_real_full, size=n_real, replace=False)
        X_real = X_real_full[idx_real]
        y_real = y_real_full[idx_real]
    else:
        X_real = X_real_full
        y_real = y_real_full

    for sigma in sigmas:
        # create noisy synthetic
        noise = rng.normal(scale=sigma, size=X_syn_base.shape)
        X_syn = X_syn_base + noise
        y_syn = y_syn_base
        for ratio in ratios:
            # build combined training set (same logic as train_mlp_and_eval)
            if ratio >= 1.0:
                X_train = X_real
                y_train = y_real
            elif ratio <= 0.0:
                X_train = X_syn
                y_train = y_syn
            else:
                n_real_local = len(y_real)
                n_syn_target = int(n_real_local * (1.0 - ratio) / (ratio + 1e-12))
                if n_syn_target <= 0 or len(y_syn) == 0:
                    X_train = X_real
                    y_train = y_real
                else:
                    idx = rng.choice(len(y_syn), size=n_syn_target, replace=(len(y_syn) < n_syn_target))
                    X_syn_s = X_syn[idx]
                    y_syn_s = y_syn[idx]
                    X_train = np.vstack([X_real, X_syn_s])
                    y_train = np.hstack([y_real, y_syn_s])

            # train MLP (small) and evaluate
            mlp_params = {'model__max_iter': int(args.mlp_max_iter)}
            model = build_mlp(params=mlp_params, random_state=args.random_state)
            model.fit(X_train, y_train)
            acc = (model.predict(X_test) == y_test).mean()
            print(f'sigma={sigma}, ratio={ratio} -> acc={acc:.6f}')
            results.append({'sigma': sigma, 'ratio': ratio, 'accuracy': float(acc), 'train_points': len(X_train)})

    out_df = pd.DataFrame(results)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print('Wrote:', args.out_csv)


if __name__ == '__main__':
    main()
