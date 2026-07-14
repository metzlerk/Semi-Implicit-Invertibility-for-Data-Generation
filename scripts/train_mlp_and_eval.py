#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

num_classes = 8
class_cols = ["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# PyTorch MLP Classifier
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
# Training loop with early stopping
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=20,
                ckpt_path="best_classifier.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1} | val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(ckpt_path))
    return model

# -----------------------------
# Data loading helpers
# -----------------------------
NON_FEATURE_COLUMNS = {"Unnamed: 0", "index", "__index_level_0__", "TemperatureKelvin", "PressureBar", "temp_K"}

def load_df(path):
    df = pd.read_feather(path)
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def df_to_xy(df):
    # Exclude bookkeeping/condition columns, the string Label, and the one-hot
    # class columns (DEB..TEPO) so the class identity never leaks into X.
    excluded = NON_FEATURE_COLUMNS | set(class_cols) | {"Label"}
    feature_cols = [c for c in df.columns if c not in excluded]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    class_to_int = {name: i for i, name in enumerate(class_cols)}

    if "Label" in df.columns:
        labels = df["Label"].astype(str).str.strip()
        if not set(labels.unique()).issubset(set(class_to_int.keys())):
            print("Unexpected labels:", set(labels.unique()) - set(class_to_int.keys()))
            raise ValueError("Found labels not in class_to_int mapping")
        y = labels.map(class_to_int).to_numpy()

    elif all(c in df.columns for c in class_cols):
        one_hot = df[class_cols].to_numpy()
        if not ((one_hot.sum(axis=1) == 1).all()):
            raise ValueError("Synthetic one-hot rows are not exactly one-hot")
        y = one_hot.argmax(axis=1)

    else:
        raise ValueError("No usable label information found in dataframe")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    return X, y, feature_cols

# -----------------------------
# Plotting helpers
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
        xticklabels=class_cols,
        yticklabels=class_cols,
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
    p = argparse.ArgumentParser()
    p.add_argument("--real-feather", required=True)
    p.add_argument("--test-feather", required=True)
    p.add_argument("--synthetic-feather", default=None,
                   help="Optional synthetic feather to mix in. Omit for a real-only baseline.")
    p.add_argument("--ratio", type=float, default=0.5,
                   help="Fraction of the training set that is real (1.0 = real only).")
    p.add_argument("--std-label", default="mlp_eval",
                   help="Run label used to name output PNGs and tag the results CSV.")
    p.add_argument("--save-model", default=None,
                   help="Path to save the trained classifier state_dict (.pt).")
    p.add_argument("--out-csv", default=None,
                   help="Path to write a one-row metrics CSV.")
    args = p.parse_args()

    label = args.std_label
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # Load real + test
    real_df = load_df(args.real_feather)
    test_df = load_df(args.test_feather)
    X_real, y_real, feature_cols = df_to_xy(real_df)
    X_test, y_test, _ = df_to_xy(test_df)

    # Optionally mix in synthetic samples (skipped for a real-only baseline)
    n_syn_used = 0
    if args.synthetic_feather and args.ratio < 1.0:
        syn_df = load_df(args.synthetic_feather)
        _, y_syn, _ = df_to_xy(syn_df)
        X_syn = syn_df[feature_cols].to_numpy(dtype=np.float32)

        n_real = len(X_real)
        n_syn_target = int(n_real * (1 - args.ratio) / args.ratio)
        idx = np.random.choice(len(X_syn), size=n_syn_target, replace=True)
        X_syn_s = X_syn[idx]
        y_syn_s = y_syn[idx]
        n_syn_used = len(X_syn_s)

        X_train = np.vstack([X_real, X_syn_s])
        y_train = np.hstack([y_real, y_syn_s])
    else:
        if args.synthetic_feather and args.ratio >= 1.0:
            print("ratio >= 1.0: ignoring --synthetic-feather (real-only run).")
        X_train = X_real
        y_train = y_real
    print(f"Training on {len(X_real)} real + {n_syn_used} synthetic = {len(X_train)} samples "
          f"({len(feature_cols)} features); testing on {len(X_test)} samples.")

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Train/val split
    val_split = int(0.1 * len(X_train))
    X_val, y_val = X_train[:val_split], y_train[:val_split]
    X_train2, y_train2 = X_train[val_split:], y_train[val_split:]

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train2, dtype=torch.float32),
            torch.tensor(y_train2, dtype=torch.long),
        ),
        batch_size=512,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=512,
    )

    # Model
    model = MLPClassifierTorch(input_dim=X_train.shape[1], num_classes=num_classes).to(device)

    # Train (best checkpoint kept per-run so concurrent labels don't collide)
    ckpt_path = os.path.join(out_dir, f"{label}_best.pt")
    model = train_model(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)) \
                    .argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.6f}")

    # -----------------------------
    # Confusion matrix + accuracy bar (labelled per run)
    # -----------------------------
    cm = confusion_matrix(y_test, preds, labels=np.arange(num_classes))

    plot_confusion(cm, acc, os.path.join(out_dir, f"{label}_confusion_norm.png"), normalized=True)
    plot_confusion(cm, acc, os.path.join(out_dir, f"{label}_confusion_raw.png"), normalized=False)
    plot_accuracy_bar(acc, os.path.join(out_dir, f"{label}_accuracy_bar.png"))

    # -----------------------------
    # Persist trained model + metrics
    # -----------------------------
    if args.save_model:
        os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved model to {args.save_model}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        pd.DataFrame([{
            "std_label": label,
            "ratio": args.ratio,
            "n_real": len(X_real),
            "n_synthetic": n_syn_used,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            "test_accuracy": acc,
        }]).to_csv(args.out_csv, index=False)
        print(f"Wrote metrics to {args.out_csv}")

if __name__ == "__main__":
    main()
