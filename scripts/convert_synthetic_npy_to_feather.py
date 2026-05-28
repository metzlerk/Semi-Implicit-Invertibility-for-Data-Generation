#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import pandas as pd


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


def main(real_feather_path, spectra_npy, labels_npy, out_feather):
    real_df = _safe_read_feather(
        real_feather_path,
        fallback_paths=[
            Path(real_feather_path).with_name('clean_split_1_of_5.feather'),
            Path(real_feather_path).with_name('clean_test_data.feather'),
            Path(real_feather_path).with_name('test_data.feather'),
        ],
    )
    # determine feature columns from real
    label_size = 8
    feature_cols = list(real_df.columns[: (real_df.shape[1] - label_size)])
    data_size = len(feature_cols)

    X = np.load(spectra_npy)
    y = np.load(labels_npy)

    # align widths: truncate or pad with zeros
    if X.shape[1] > data_size:
        X = X[:, :data_size]
    elif X.shape[1] < data_size:
        pad = np.zeros((X.shape[0], data_size - X.shape[1]), dtype=X.dtype)
        X = np.hstack([X, pad])

    df_feats = pd.DataFrame(X, columns=feature_cols)

    # build one-hot label columns
    CLASS_NAMES = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    one_hot = np.zeros((len(y), len(CLASS_NAMES)), dtype=int)
    for i, lab in enumerate(y):
        if 0 <= int(lab) < len(CLASS_NAMES):
            one_hot[i, int(lab)] = 1
    df_labels = pd.DataFrame(one_hot, columns=CLASS_NAMES)

    out_df = pd.concat([df_feats.reset_index(drop=True), df_labels.reset_index(drop=True)], axis=1)
    out_df.to_feather(out_feather)
    print(f'Wrote {out_feather} (features={data_size}, rows={len(out_df)})')


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: convert_synthetic_npy_to_feather.py <real_feather> <spectra_npy> <labels_npy> <out_feather>')
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
