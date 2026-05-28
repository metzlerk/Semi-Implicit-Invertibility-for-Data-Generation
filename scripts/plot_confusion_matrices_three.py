#!/usr/bin/env python3
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


CLASS_NAMES = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
NON_FEATURE_COLUMNS = {'Unnamed: 0', 'index', 'Label', 'TemperatureKelvin', 'PressureBar', 'temp_K'}


def load_frame(path):
    df = pd.read_feather(path)
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    label_size = len(CLASS_NAMES)
    if df.shape[1] < label_size:
        raise ValueError(f'{path} does not contain enough label columns after dropping metadata')
    X = df.iloc[:, : df.shape[1] - label_size].values
    y = df.iloc[:, -label_size:].values.argmax(axis=1)
    return X, y


def load_case(name, model_path, test_path):
    model = joblib.load(model_path)
    X_test, y_test = load_frame(test_path)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(CLASS_NAMES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    return {
        'name': name,
        'model_path': model_path,
        'test_path': test_path,
        'accuracy': acc,
        'cm': cm,
        'cm_norm': cm_norm,
    }


def plot_cases(cases, out_path, normalized=True):
    sns.set_style('white')
    fig, axes = plt.subplots(1, len(cases), figsize=(6.2 * len(cases), 6), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]

    vmax = 1.0 if normalized else max(case['cm'].max() for case in cases)
    cmap = 'Blues'

    for ax, case in zip(axes, cases):
        matrix = case['cm_norm'] if normalized else case['cm']
        if normalized:
            annot = np.vectorize(lambda x: f'{x:.2f}')(matrix)
            title_suffix = f"acc={case['accuracy']:.6f}"
        else:
            annot = case['cm']
            title_suffix = f"acc={case['accuracy']:.6f}"

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            annot=annot,
            fmt='',
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            square=True,
            linewidths=0.5,
            linecolor='white',
        )
        ax.set_title(f"{case['name']}\n{title_suffix}", pad=14)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    fig.suptitle('Confusion Matrices for Three MLP Experiments', y=1.03, fontsize=16)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Wrote {out_path}')


def main():
    repo = Path('/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation')
    cases = [
        load_case(
            'Real-only (full)',
            repo / 'models/mlp_realonly_full.joblib',
            repo / 'Data/test_data.feather',
        ),
        load_case(
            'Selected std1.5 r=0.6',
            repo / 'models/mlp_std1.5_r0.6000000000000001.joblib',
            repo / 'Data/test_data.feather',
        ),
        load_case(
            'Temp-split (bottom80)',
            repo / 'models/mlp_realonly_temp_split.joblib',
            Path('/home/kjmetzler/scratch/test_temp_top20.feather'),
        ),
    ]

    out_dir = repo / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cases(cases, out_dir / 'confusion_matrices_three_experiments.png', normalized=True)

    for case in cases:
        single_out = out_dir / f"confusion_{case['name'].lower().replace(' ', '_').replace('/', '_')}.png"
        plot_cases([case], single_out, normalized=True)


if __name__ == '__main__':
    main()
