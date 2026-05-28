#!/usr/bin/env python3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CLASS_NAMES = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']


def load_frame(path):
    df = pd.read_feather(path)
    drop_cols = [c for c in {'Unnamed: 0', 'index', 'Label', 'TemperatureKelvin', 'PressureBar', 'temp_K'} if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    label_size = len(CLASS_NAMES)
    X = df.iloc[:, : df.shape[1] - label_size].values
    y = df.iloc[:, -label_size:].values.argmax(axis=1)
    return X, y


def load_balanced_sample(path, per_class=500, random_state=42):
    X, y = load_frame(path)
    rng = np.random.default_rng(random_state)
    indices = []
    for class_label in range(len(CLASS_NAMES)):
        class_indices = np.where(y == class_label)[0]
        if len(class_indices) == 0:
            continue
        take = min(per_class, len(class_indices))
        indices.append(rng.choice(class_indices, size=take, replace=False))
    indices = np.concatenate(indices)
    return X[indices], y[indices]


def load_accuracy(csv_path):
    return float(pd.read_csv(csv_path)['accuracy'].iloc[0])


def evaluate_model(model_path, X_test, y_test, accuracy):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(CLASS_NAMES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    return {'model_path': model_path, 'accuracy': accuracy, 'cm': cm, 'cm_norm': cm_norm}


def plot_cases(cases, out_path, normalized=True):
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    vmax = 1.0 if normalized else max(case['cm'].max() for case in cases)

    for ax, case in zip(axes, cases):
        matrix = case['cm_norm'] if normalized else case['cm']
        image = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=vmax)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                text = f'{matrix[row, col]:.2f}' if normalized else str(int(matrix[row, col]))
                ax.text(col, row, text, ha='center', va='center', fontsize=7, color='black')

        ax.set_xticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlim(-0.5, len(CLASS_NAMES) - 0.5)
        ax.set_ylim(len(CLASS_NAMES) - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f"{Path(case['model_path']).stem}\nacc={case['accuracy']:.4f}", pad=14)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='both', length=0)

    fig.colorbar(image, ax=axes, shrink=0.85, pad=0.01)

    fig.suptitle('Confusion Matrices for 5 MLP Experiments', y=1.03, fontsize=16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Wrote {out_path}')


def plot_accuracy_bars(cases, out_path):
    names = [Path(case['model_path']).stem.replace('mlp_', '').replace('_', ' ') for case in cases]
    accuracies = [case['accuracy'] for case in cases]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(cases)))
    bars = ax.bar(names, accuracies, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier Accuracy by Split')
    ax.tick_params(axis='x', rotation=20)
    ax.bar_label(bars, labels=[f'{acc:.4f}' for acc in accuracies], padding=3, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Wrote {out_path}')


def main():
    repo = Path('.').resolve()
    test_path = repo / 'Data' / 'test_data.feather'
    X_test, y_test = load_balanced_sample(test_path, per_class=500, random_state=42)
    cases = []
    for i in range(1, 6):
        model_path = repo / f'models/mlp_split_{i}.joblib'
        csv_path = repo / f'results/mlp_split_{i}_results.csv'
        if not model_path.exists():
            print(f'model missing: {model_path}; skipping')
            continue
        cases.append(evaluate_model(model_path, X_test, y_test, load_accuracy(csv_path)))

    if not cases:
        print('No models found to plot. Exiting.')
        return

    out = repo / 'results' / 'confusion_matrices_5_splits.png'
    plot_cases(cases, out, normalized=True)

    bar_out = repo / 'results' / 'classifier_accuracy_bars_5_splits.png'
    plot_accuracy_bars(cases, bar_out)


if __name__ == '__main__':
    main()
