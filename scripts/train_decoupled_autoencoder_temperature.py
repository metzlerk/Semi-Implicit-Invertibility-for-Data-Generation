#!/usr/bin/env python3
"""Train a decoupled IMS autoencoder with a 513-d latent target.

The encoder learns to predict a 512-d ChemNet embedding plus a temperature
scalar in the 513th latent position. The decoder reconstructs the 1,676 IMS
spectral inputs from that full latent code.
"""

import argparse
import ast
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = "/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation"
DATA_DIR = os.path.join(ROOT_DIR, "Data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a decoupled IMS autoencoder.")
    parser.add_argument(
        "--train-data",
        default=os.path.join(DATA_DIR, "train_data_with_conditions.feather"),
        help="Training feather file containing IMS spectra and temperature.",
    )
    parser.add_argument(
        "--embedding-file",
        default=os.path.join(DATA_DIR, "name_smiles_embedding_file.csv"),
        help="ChemNet embedding CSV file.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(MODELS_DIR, "decoupled_autoencoder_temperature.pth"),
        help="Output checkpoint path.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-layers", type=int, default=9)
    parser.add_argument("--latent-dim", type=int, default=513)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--chem-weight", type=float, default=1.0)
    parser.add_argument("--temp-weight", type=float, default=1.0)
    parser.add_argument("--temperature-scale", type=float, default=300.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="ims_encoder_decoder")
    parser.add_argument("--wandb-run-name", type=str, default="decoupled_autoencoder_temperature")
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def find_column(df, candidates):
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        candidate_lower = candidate.lower()
        if candidate_lower in lower_map:
            return lower_map[candidate_lower]
    raise ValueError(f"Could not find any of {candidates} in columns: {list(df.columns)[:12]}...")


def load_feature_matrix(feather_path):
    df = pd.read_feather(feather_path)
    feature_cols = [col for col in df.columns if col.startswith(("p", "n"))]
    if len(feature_cols) != 1676:
        raise ValueError(f"Expected 1676 p/n feature columns, found {len(feature_cols)}.")

    temp_col = find_column(df, ["TemperatureKelvin", "temperatureKelvin", "temperature_kelvin", "temperature"])
    label_col = find_column(df, ["Label"])

    spectra = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    temperature = df[temp_col].to_numpy(dtype=np.float32, copy=False)
    labels = df[label_col].astype(str).to_numpy()
    return spectra, temperature, labels, feature_cols, temp_col


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


class DecoupledAutoencoder(nn.Module):
    def __init__(self, input_dim=1676, latent_dim=513, n_layers=9):
        super().__init__()
        encoder_sizes = np.linspace(input_dim, latent_dim, n_layers + 1).astype(int).tolist()
        decoder_sizes = np.linspace(latent_dim, input_dim, n_layers + 1).astype(int).tolist()

        encoder_layers = []
        for idx in range(len(encoder_sizes) - 1):
            encoder_layers.append(nn.Linear(encoder_sizes[idx], encoder_sizes[idx + 1]))
            if idx < len(encoder_sizes) - 2:
                encoder_layers.append(nn.LeakyReLU(0.1, inplace=True))

        decoder_layers = []
        for idx in range(len(decoder_sizes) - 1):
            decoder_layers.append(nn.Linear(decoder_sizes[idx], decoder_sizes[idx + 1]))
            if idx < len(decoder_sizes) - 2:
                decoder_layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def make_loader(spectra, chem_targets, temp_targets, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(spectra),
        torch.from_numpy(chem_targets),
        torch.from_numpy(temp_targets),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)


def evaluate(model, loader, device, mse, recon_weight, chem_weight, temp_weight):
    model.eval()
    total = 0.0
    recon_total = 0.0
    chem_total = 0.0
    temp_total = 0.0
    count = 0

    with torch.no_grad():
        for spectra_batch, chem_batch, temp_batch in loader:
            spectra_batch = spectra_batch.to(device)
            chem_batch = chem_batch.to(device)
            temp_batch = temp_batch.to(device)

            recon, latent = model(spectra_batch)
            recon_loss = mse(recon, spectra_batch)
            chem_loss = mse(latent[:, :512], chem_batch)
            temp_loss = mse(latent[:, 512], temp_batch)
            loss = recon_weight * recon_loss + chem_weight * chem_loss + temp_weight * temp_loss

            batch_size = spectra_batch.size(0)
            total += loss.item() * batch_size
            recon_total += recon_loss.item() * batch_size
            chem_total += chem_loss.item() * batch_size
            temp_total += temp_loss.item() * batch_size
            count += batch_size

    return {
        "loss": total / count,
        "recon": recon_total / count,
        "chem": chem_total / count,
        "temp": temp_total / count,
    }


def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if "SLURM_JOB_ID" not in os.environ:
        sys.stderr.write(
            "ERROR: This training script must be submitted via SLURM.\n"
            "Submit with: sbatch scripts/run_train_decoupled_autoencoder_temperature.sh\n"
        )
        raise SystemExit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        import wandb
    except Exception:
        wandb = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    spectra, temperature, labels, feature_cols, temp_col = load_feature_matrix(args.train_data)
    embedding_lookup = load_chemnet_embeddings(args.embedding_file)

    missing_labels = sorted(set(labels) - set(embedding_lookup.keys()))
    if missing_labels:
        raise ValueError(f"Missing ChemNet embeddings for labels: {missing_labels}")

    chem_targets = np.stack([embedding_lookup[label] for label in labels]).astype(np.float32)
    if chem_targets.shape[1] != 512:
        raise ValueError(f"Expected 512-d ChemNet targets, found {chem_targets.shape[1]}.")

    temp_targets = (temperature / float(args.temperature_scale)).astype(np.float32)

    n_samples = len(spectra)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(n_samples * args.val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if args.max_train_samples and args.max_train_samples > 0:
        train_indices = train_indices[: min(len(train_indices), args.max_train_samples)]
    if args.max_val_samples and args.max_val_samples > 0:
        val_indices = val_indices[: min(len(val_indices), args.max_val_samples)]

    train_loader = make_loader(
        spectra[train_indices],
        chem_targets[train_indices],
        temp_targets[train_indices],
        args.batch_size,
        shuffle=True,
    )
    val_loader = make_loader(
        spectra[val_indices],
        chem_targets[val_indices],
        temp_targets[val_indices],
        args.batch_size,
        shuffle=False,
    )

    model = DecoupledAutoencoder(input_dim=1676, latent_dim=args.latent_dim, n_layers=args.n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()

    wandb_run = None
    if wandb is not None and not args.no_wandb and os.environ.get("WANDB_MODE", "").lower() != "disabled":
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "train_data": args.train_data,
                    "embedding_file": args.embedding_file,
                    "latent_dim": args.latent_dim,
                    "n_layers": args.n_layers,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "recon_weight": args.recon_weight,
                    "chem_weight": args.chem_weight,
                    "temp_weight": args.temp_weight,
                    "temperature_scale": args.temperature_scale,
                    "val_fraction": args.val_fraction,
                    "seed": args.seed,
                },
            )
        except Exception as exc:
            print(f"W&B init failed, continuing without logging: {exc}")
            wandb_run = None

    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_chem = 0.0
        train_temp = 0.0
        train_count = 0

        for spectra_batch, chem_batch, temp_batch in train_loader:
            spectra_batch = spectra_batch.to(device, non_blocking=True)
            chem_batch = chem_batch.to(device, non_blocking=True)
            temp_batch = temp_batch.to(device, non_blocking=True)

            recon, latent = model(spectra_batch)
            recon_loss = mse(recon, spectra_batch)
            chem_loss = mse(latent[:, :512], chem_batch)
            temp_loss = mse(latent[:, 512], temp_batch)
            loss = args.recon_weight * recon_loss + args.chem_weight * chem_loss + args.temp_weight * temp_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = spectra_batch.size(0)
            train_total += loss.item() * batch_size
            train_recon += recon_loss.item() * batch_size
            train_chem += chem_loss.item() * batch_size
            train_temp += temp_loss.item() * batch_size
            train_count += batch_size

        train_metrics = {
            "loss": train_total / train_count,
            "recon": train_recon / train_count,
            "chem": train_chem / train_count,
            "temp": train_temp / train_count,
        }
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            mse,
            args.recon_weight,
            args.chem_weight,
            args.temp_weight,
        )

        history.append({"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} | "
                f"recon={val_metrics['recon']:.6f} chem={val_metrics['chem']:.6f} temp={val_metrics['temp']:.6f}"
            )

        if wandb_run is not None:
            wandb.log({"epoch": epoch + 1, **{f"train/{k}": v for k, v in train_metrics.items()}, **{f"val/{k}": v for k, v in val_metrics.items()}})

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val,
                    "feature_columns": feature_cols,
                    "temperature_column": temp_col,
                    "temperature_scale": args.temperature_scale,
                    "latent_dim": args.latent_dim,
                    "n_layers": args.n_layers,
                    "loss_weights": {
                        "recon": args.recon_weight,
                        "chem": args.chem_weight,
                        "temp": args.temp_weight,
                    },
                },
                args.out,
            )

    metrics_path = os.path.splitext(args.out)[0] + "_history.csv"
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    print(f"Saved best checkpoint -> {args.out} (val loss={best_val:.6f})")
    print(f"Saved training history -> {metrics_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()