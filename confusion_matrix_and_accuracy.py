#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# -----------------------------
# Constants
# -----------------------------
CLASS_NAMES = ["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]
NON_FEATURE_COLUMNS = {"Unnamed: 0", "index", "__index_level_0__", "TemperatureKelvin", "PressureBar", "temp_K"}

# -----------------------------
# Model definition (same as training)
# -----------------------------
class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Data loading
# -----------------------------
def load_df(path):
    df = pd.read_feather(path)
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def df_to_xy(df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    class_to_int = {name: i for i, name in enumerate(CLASS_NAMES)}

    if "Label" in df.columns:
        labels = df["Label"].astype(str).str.strip()
        y = labels.map(class_to_int).to_numpy()

    elif all(c in df.columns for c in CLASS_NAMES):
        onehot = df[CLASS_NAMES].to_numpy()
        y = onehot.argmax(axis=1)

    else:
        raise ValueError("No usable label information found in dataframe")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    return X, y, feature_cols

# -----------------------------
# Plot confusion matrix
# -----------------------------
def plot_confusion(cm, acc, out_path, normalized=True):
    sns.set_style("white")

    if normalized:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        annot = np.vectorize(lambda x: f"{x:.2f}")(cm_plot)
        vmax = 1.0
    else:
        cm_plot = cm
        annot = cm
        vmax = cm.max()

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_plot,
        cmap="Blues",
        vmin=0,
        vmax=vmax,
        annot=annot,
        fmt="",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix\nAccuracy = {acc:.6f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

# -----------------------------
# Plot accuracy bar graph
# -----------------------------
def plot_accuracy_bar(acc, out_path):
    plt.figure(figsize=(5, 6))
    sns.barplot(x=["Accuracy"], y=[acc], color="steelblue")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Overall Accuracy = {acc:.6f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-feather", required=True)
    parser.add_argument("--out-prefix", default="results")
    args = parser.parse_args()

    # Load test data
    test_df = load_df(args.test_feather)
    X_test, y_test, feature_cols = df_to_xy(test_df)

    # Load model
    input_dim = X_test.shape[1]
    model = MLPClassifierTorch(input_dim=input_dim, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()

    # Metrics
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=np.arange(len(CLASS_NAMES)))

    # Outputs
    plot_confusion(cm, acc, f"{args.out_prefix}_confusion_norm.png", normalized=True)
    plot_confusion(cm, acc, f"{args.out_prefix}_confusion_raw.png", normalized=False)
    plot_accuracy_bar(acc, f"{args.out_prefix}_accuracy_bar.png")

if __name__ == "__main__":
    main()
