#!/usr/bin/env python3
import argparse
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Constants
# -----------------------------
CLASS_NAMES = ["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]
NON_FEATURE_COLUMNS = {"Unnamed: 0", "index", "__index_level_0__", "PressureBar", "temp_K"}
TEMP_COL = "TemperatureKelvin"
TEMP_SCALE = 300.0

def get_device():
    if torch.cuda.is_available():
        try:
            test_device = torch.device("cuda")
            _ = torch.zeros(1, device=test_device)
            return test_device
        except Exception as exc:
            print(f"CUDA is available but unusable with this PyTorch build; falling back to CPU: {exc}")
    return torch.device("cpu")


device = get_device()
print("Using device:", device)

# -----------------------------
# Models
# -----------------------------
class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Helpers
# -----------------------------
def load_df(path):
    df = pd.read_feather(path)
    drop_cols = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df


def df_to_xy(df):
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "Label"]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    class_to_int = {name: i for i, name in enumerate(CLASS_NAMES)}

    if "Label" in df.columns:
        labels = df["Label"].astype(str).str.strip()
        if not set(labels.unique()).issubset(set(class_to_int.keys())):
            print("Unexpected labels:", set(labels.unique()) - set(class_to_int.keys()))
            raise ValueError("Found labels not in class_to_int mapping")
        y = labels.map(class_to_int).to_numpy()
    elif all(c in df.columns for c in CLASS_NAMES):
        onehot = df[CLASS_NAMES].to_numpy()
        if not ((onehot.sum(axis=1) == 1).all()):
            raise ValueError("Synthetic one-hot rows are not exactly one-hot")
        y = onehot.argmax(axis=1)
    else:
        raise ValueError("No usable label information found in dataframe")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    return X, y, feature_cols


def load_chemnet_embeddings(embedding_file):
    embedding_df = pd.read_csv(embedding_file)
    if "embedding" not in embedding_df.columns:
        raise ValueError("Expected an 'embedding' column in the ChemNet CSV.")

    embedding_lookup = {}
    for _, row in embedding_df.iterrows():
        embedding_raw = row.get("embedding")
        if pd.isna(embedding_raw) or embedding_raw == "":
            continue
        embedding = np.asarray(ast.literal_eval(embedding_raw), dtype=np.float32)
        short_name = row.get("Unnamed: 0")
        long_name = row.get("Name")
        if pd.notna(short_name):
            embedding_lookup[str(short_name)] = embedding
        if pd.notna(long_name):
            embedding_lookup[str(long_name)] = embedding

    return embedding_lookup


def build_latent_targets(labels, temperatures, embedding_lookup, temperature_scale=TEMP_SCALE):
    missing_labels = sorted(set(labels) - set(embedding_lookup.keys()))
    if missing_labels:
        raise ValueError(f"Missing ChemNet embeddings for labels: {missing_labels}")

    chem_targets = np.stack([embedding_lookup[label] for label in labels]).astype(np.float32)
    if chem_targets.shape[1] != 512:
        raise ValueError(f"Expected 512-d ChemNet targets, found {chem_targets.shape[1]}.")

    temp_targets = (temperatures / float(temperature_scale)).astype(np.float32)
    return chem_targets, temp_targets


def make_temperature_sandwich(df, bottom_frac=0.5, middle_frac=0.2, top_frac=0.3):
    assert abs(bottom_frac + middle_frac + top_frac - 1.0) < 1e-6
    df_sorted = df.sort_values(TEMP_COL).reset_index(drop=True)
    n = len(df_sorted)
    n_bottom = int(bottom_frac * n)
    n_middle = int(middle_frac * n)
    n_top = n - n_bottom - n_middle

    bottom_slice = df_sorted.iloc[:n_bottom]
    middle_slice = df_sorted.iloc[n_bottom:n_bottom + n_middle]
    top_slice = df_sorted.iloc[n_bottom + n_middle:]

    bread_df = pd.concat([bottom_slice, top_slice], ignore_index=True)
    ham_df = middle_slice.copy()

    print(f"Sandwich split: bottom={n_bottom}, middle={n_middle}, top={n_top}")
    return bread_df, ham_df


def train_autoencoder(
    encoder,
    decoder,
    train_loader,
    val_loader,
    epochs=200,
    lr=1e-3,
    patience=20,
    recon_weight=1.0,
    chem_weight=1.0,
    temp_weight=1.0,
):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        for xb, chem_batch, temp_batch in train_loader:
            xb = xb.to(device)
            chem_batch = chem_batch.to(device)
            temp_batch = temp_batch.to(device)
            optimizer.zero_grad()
            latent = encoder(xb)
            preds = decoder(latent)
            recon_loss = criterion(preds, xb)
            chem_loss = criterion(latent[:, :512], chem_batch)
            temp_loss = criterion(latent[:, 512], temp_batch)
            loss = recon_weight * recon_loss + chem_weight * chem_loss + temp_weight * temp_loss
            loss.backward()
            optimizer.step()

        encoder.eval()
        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for xb, chem_batch, temp_batch in val_loader:
                xb = xb.to(device)
                chem_batch = chem_batch.to(device)
                temp_batch = temp_batch.to(device)
                latent = encoder(xb)
                preds = decoder(latent)
                recon_loss = criterion(preds, xb)
                chem_loss = criterion(latent[:, :512], chem_batch)
                temp_loss = criterion(latent[:, 512], temp_batch)
                loss = recon_weight * recon_loss + chem_weight * chem_loss + temp_weight * temp_loss
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f"[AE] Epoch {epoch+1} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, "best_autoencoder.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[AE] Early stopping.")
                break

    checkpoint = torch.load("best_autoencoder.pt")
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    return encoder, decoder


def train_classifier(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f"[CLS] Epoch {epoch+1} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_classifier_sandwich.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[CLS] Early stopping.")
                break

    model.load_state_dict(torch.load("best_classifier_sandwich.pt"))
    return model


def make_confusion_and_bar(y_true, y_pred, name_prefix):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    print(f"[{name_prefix}] Accuracy: {acc:.6f}")

    sns.set_style("white")

    # Normalized confusion
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    annot_norm = np.vectorize(lambda x: f"{x:.2f}")(cm_norm)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        cmap="Blues",
        vmin=0,
        vmax=1.0,
        annot=annot_norm,
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
    plt.title(f"{name_prefix} Confusion (norm)\nAcc={acc:.6f}")
    plt.tight_layout()
    plt.savefig(f"{name_prefix}_confusion_norm.png", dpi=200)
    plt.close()

    # Raw confusion
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        cmap="Blues",
        vmin=0,
        vmax=cm.max(),
        annot=cm,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name_prefix} Confusion (raw)\nAcc={acc:.6f}")
    plt.tight_layout()
    plt.savefig(f"{name_prefix}_confusion_raw.png", dpi=200)
    plt.close()

    # Accuracy bar
    plt.figure(figsize=(5, 6))
    sns.barplot(x=["Accuracy"], y=[acc], color="steelblue")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"{name_prefix} Accuracy = {acc:.6f}")
    plt.tight_layout()
    plt.savefig(f"{name_prefix}_accuracy_bar.png", dpi=200)
    plt.close()


def generate_synthetic_dmmp(
    bread_df,
    ham_df,
    encoder,
    decoder,
    feature_cols,
    n_samples=5000,
    temperature_scale=TEMP_SCALE,
):
    # Use DMMP from bread as source
    dmmp_bread = bread_df[bread_df["Label"] == "DMMP"].copy()
    if dmmp_bread.empty:
        print("No DMMP in bread slice; cannot generate synthetic DMMP.")
        return pd.DataFrame()

    X_bread_dmmp = dmmp_bread[feature_cols].to_numpy(dtype=np.float32)
    temps_ham = ham_df[TEMP_COL].to_numpy()
    temp_low, temp_high = temps_ham.min(), temps_ham.max()
    print(f"Synthetic DMMP temp range: [{temp_low:.2f}, {temp_high:.2f}]")

    # Normalize like AE training
    X_mean = X_bread_dmmp.mean(axis=0, keepdims=True)
    X_std = X_bread_dmmp.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_bread_dmmp - X_mean) / X_std

    temp_idx = feature_cols.index(TEMP_COL)
    temp_target = X_bread_dmmp[:, temp_idx].reshape(-1, 1)
    temp_norm = (temp_target / float(temperature_scale)).astype(np.float32)
    X_norm[:, temp_idx] = temp_norm[:, 0]

    with torch.no_grad():
        Z = encoder(torch.tensor(X_norm, dtype=torch.float32).to(device)).cpu().numpy()

    # Latent dims
    latent_dim = Z.shape[1]
    temp_latent_idx = latent_dim - 1  # assume last dim encodes temp like before

    Z_dmmp = Z
    latent_std = np.std(Z_dmmp[:, :latent_dim - 1], axis=0)
    perturb_std = 0.1 * np.mean(latent_std)

    base_indices = np.random.choice(len(Z_dmmp), size=n_samples, replace=True)
    base_latent = Z_dmmp[base_indices].copy()

    noise = np.random.normal(0, perturb_std, size=(n_samples, latent_dim - 1))
    base_latent[:, :latent_dim - 1] += noise

    new_temps_phys = np.random.uniform(temp_low, temp_high, size=(n_samples, 1))
    new_temps_norm = new_temps_phys / float(temperature_scale)
    base_latent[:, temp_latent_idx] = new_temps_norm[:, 0]

    decoded_batches = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_latent = torch.tensor(base_latent[start:end], dtype=torch.float32).to(device)
            decoded = decoder(batch_latent).cpu().numpy()
            decoded_batches.append(decoded)

    decoded_norm = np.vstack(decoded_batches)
    decoded_phys = decoded_norm * X_std + X_mean

    temp_decoded = decoded_phys[:, temp_idx]
    valid_mask = (
        (~np.isnan(decoded_phys).any(axis=1)) &
        (~np.isinf(decoded_phys).any(axis=1)) &
        (temp_decoded >= temp_low) &
        (temp_decoded <= temp_high)
    )

    decoded_phys = decoded_phys[valid_mask]
    new_temps_phys = new_temps_phys[valid_mask]

    rows = []
    for i in range(decoded_phys.shape[0]):
        row = {"Label": "DMMP"}
        for j, col in enumerate(feature_cols):
            if col == TEMP_COL:
                row[col] = float(new_temps_phys[i, 0])
            else:
                row[col] = float(decoded_phys[i, j])
        rows.append(row)

    syn_df = pd.DataFrame(rows)
    print(f"Generated {len(syn_df)} synthetic DMMP samples.")
    return syn_df


def make_pca_3d_plot(bread_X, ham_X, out_path):
    X_all = np.vstack([bread_X, ham_X])
    labels = np.array([0] * len(bread_X) + [1] * len(ham_X))

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_all)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    bread_pts = X_pca[labels == 0]
    ham_pts = X_pca[labels == 1]

    ax.scatter(bread_pts[:, 0], bread_pts[:, 1], bread_pts[:, 2], s=5, alpha=0.6, label="Bread")
    ax.scatter(ham_pts[:, 0], ham_pts[:, 1], ham_pts[:, 2], s=5, alpha=0.6, label="Ham")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-feather", default="Data/train_data_with_conditions.feather")
    parser.add_argument("--embedding-file", default="Data/name_smiles_embedding_file.csv")
    parser.add_argument("--bottom-frac", type=float, default=0.5)
    parser.add_argument("--middle-frac", type=float, default=0.2)
    parser.add_argument("--top-frac", type=float, default=0.3)
    parser.add_argument("--ham-chem", default="DMMP")
    parser.add_argument("--temperature-scale", type=float, default=TEMP_SCALE)
    args = parser.parse_args()

    df = load_df(args.data_feather)
    if TEMP_COL not in df.columns:
        raise ValueError(f"{TEMP_COL} not in dataframe.")

    # Sandwich split
    bread_df, ham_df = make_temperature_sandwich(
        df, bottom_frac=args.bottom_frac, middle_frac=args.middle_frac, top_frac=args.top_frac
    )

    # -----------------------------
    # Train encoder/decoder on bread
    # -----------------------------
    X_bread, _, feature_cols = df_to_xy(bread_df)
    embedding_lookup = load_chemnet_embeddings(args.embedding_file)
    chem_targets, temp_targets = build_latent_targets(
        bread_df["Label"].astype(str).str.strip().to_numpy(),
        bread_df[TEMP_COL].to_numpy(dtype=np.float32),
        embedding_lookup,
        temperature_scale=args.temperature_scale,
    )
    input_dim = X_bread.shape[1]
    encoder_dims = [input_dim, 1400, 1200, 1000, 800, 600, 513]
    decoder_dims = [513, 600, 800, 1000, 1200, 1400, input_dim]

    encoder = MLP(encoder_dims).to(device)
    decoder = MLP(decoder_dims).to(device)

    # Autoencoder training data: bread
    X_mean = X_bread.mean(axis=0, keepdims=True)
    X_std = X_bread.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X_bread - X_mean) / X_std

    temp_idx = feature_cols.index(TEMP_COL)
    temp_target = X_bread[:, temp_idx].reshape(-1, 1)
    temp_mean = temp_target.mean()
    temp_std = temp_target.std() + 1e-8
    temp_norm = (temp_target - temp_mean) / temp_std
    X_norm[:, temp_idx] = temp_norm[:, 0]

    # AE train/val split
    ae_val_split = int(0.1 * len(X_norm))
    X_ae_val = X_norm[:ae_val_split]
    X_ae_train = X_norm[ae_val_split:]

    chem_train = chem_targets[ae_val_split:]
    chem_val = chem_targets[:ae_val_split]
    temp_train = temp_targets[ae_val_split:]
    temp_val = temp_targets[:ae_val_split]

    ae_train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_ae_train, dtype=torch.float32),
            torch.tensor(chem_train, dtype=torch.float32),
            torch.tensor(temp_train, dtype=torch.float32),
        ),
        batch_size=512,
        shuffle=True,
    )
    ae_val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_ae_val, dtype=torch.float32),
            torch.tensor(chem_val, dtype=torch.float32),
            torch.tensor(temp_val, dtype=torch.float32),
        ),
        batch_size=512,
    )

    print("Training encoder/decoder on bread...")
    encoder, decoder = train_autoencoder(
        encoder,
        decoder,
        ae_train_loader,
        ae_val_loader,
        recon_weight=1.0,
        chem_weight=1.0,
        temp_weight=1.0,
    )

    # -----------------------------
    # Prepare classifier data
    # -----------------------------
    X_bread_cls, y_bread_cls, feature_cols_cls = df_to_xy(bread_df)
    X_ham_cls, y_ham_cls, _ = df_to_xy(ham_df)

    scaler = StandardScaler()
    X_bread_scaled = scaler.fit_transform(X_bread_cls).astype(np.float32)
    X_ham_scaled = scaler.transform(X_ham_cls).astype(np.float32)

    # Common train/val split for bread
    val_split = int(0.1 * len(X_bread_scaled))
    X_bread_val = X_bread_scaled[:val_split]
    y_bread_val = y_bread_cls[:val_split]
    X_bread_train = X_bread_scaled[val_split:]
    y_bread_train = y_bread_cls[val_split:]

    def make_cls_loaders(X_train, y_train, X_val, y_val):
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=512,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=512,
        )
        return train_loader, val_loader

    # -----------------------------
    # Task 1: bread -> ham
    # -----------------------------
    print("\n[Task 1] Train on bread, test on ham")
    cls1 = MLPClassifierTorch(input_dim=X_bread_scaled.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    train_loader1, val_loader1 = make_cls_loaders(X_bread_train, y_bread_train, X_bread_val, y_bread_val)
    cls1 = train_classifier(cls1, train_loader1, val_loader1)

    cls1.eval()
    with torch.no_grad():
        preds1 = cls1(torch.tensor(X_ham_scaled, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
    make_confusion_and_bar(y_ham_cls, preds1, "task1_bread_to_ham")

    # -----------------------------
    # Task 2: bread + one ham chem (e.g. DMMP) -> remaining ham
    # -----------------------------
    print("\n[Task 2] Train on bread + ham DMMP, test on ham without DMMP")
    ham_chem = args.ham_chem
    ham_chem_mask = ham_df["Label"] == ham_chem
    ham_chem_df = ham_df[ham_chem_mask].copy()
    ham_rest_df = ham_df[~ham_chem_mask].copy()

    X_ham_chem, y_ham_chem, _ = df_to_xy(ham_chem_df)
    X_ham_rest, y_ham_rest, _ = df_to_xy(ham_rest_df)

    # Scale using bread scaler
    X_ham_chem_scaled = scaler.transform(X_ham_chem).astype(np.float32)
    X_ham_rest_scaled = scaler.transform(X_ham_rest).astype(np.float32)

    # New train set: bread + ham_chem
    X_train2_full = np.vstack([X_bread_scaled, X_ham_chem_scaled])
    y_train2_full = np.hstack([y_bread_cls, y_ham_chem])

    val_split2 = int(0.1 * len(X_train2_full))
    X_val2 = X_train2_full[:val_split2]
    y_val2 = y_train2_full[:val_split2]
    X_train2 = X_train2_full[val_split2:]
    y_train2 = y_train2_full[val_split2:]

    train_loader2, val_loader2 = make_cls_loaders(X_train2, y_train2, X_val2, y_val2)
    cls2 = MLPClassifierTorch(input_dim=X_bread_scaled.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    cls2 = train_classifier(cls2, train_loader2, val_loader2)

    cls2.eval()
    with torch.no_grad():
        preds2 = cls2(torch.tensor(X_ham_rest_scaled, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
    make_confusion_and_bar(y_ham_rest, preds2, "task2_bread_plus_dmmp_to_ham_rest")

    # -----------------------------
    # Task 3: bread + synthetic DMMP -> ham
    # -----------------------------
    print("\n[Task 3] Train on bread + synthetic DMMP, test on ham")
    syn_dmmp_df = generate_synthetic_dmmp(
        bread_df,
        ham_df,
        encoder,
        decoder,
        feature_cols,
        temperature_scale=args.temperature_scale,
    )
    if not syn_dmmp_df.empty:
        X_syn_dmmp, y_syn_dmmp, _ = df_to_xy(syn_dmmp_df)
        X_syn_dmmp_scaled = scaler.transform(X_syn_dmmp).astype(np.float32)

        X_train3_full = np.vstack([X_bread_scaled, X_syn_dmmp_scaled])
        y_train3_full = np.hstack([y_bread_cls, y_syn_dmmp])

        val_split3 = int(0.1 * len(X_train3_full))
        X_val3 = X_train3_full[:val_split3]
        y_val3 = y_train3_full[:val_split3]
        X_train3 = X_train3_full[val_split3:]
        y_train3 = y_train3_full[val_split3:]

        train_loader3, val_loader3 = make_cls_loaders(X_train3, y_train3, X_val3, y_val3)
        cls3 = MLPClassifierTorch(input_dim=X_bread_scaled.shape[1], num_classes=len(CLASS_NAMES)).to(device)
        cls3 = train_classifier(cls3, train_loader3, val_loader3)

        cls3.eval()
        with torch.no_grad():
            preds3 = cls3(torch.tensor(X_ham_scaled, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
        make_confusion_and_bar(y_ham_cls, preds3, "task3_bread_plus_synth_dmmp_to_ham")
    else:
        print("Skipping Task 3: no synthetic DMMP generated.")

    # -----------------------------
    # Task 4: 3D PCA plot in ChemNet embedding space (bread vs ham)
    # -----------------------------
    print("\n[Task 4] 3D PCA plot of bread vs ham (ChemNet feature space)")
    make_pca_3d_plot(X_bread_cls, X_ham_cls, "bread_ham_pca_3d.png")


if __name__ == "__main__":
    main()