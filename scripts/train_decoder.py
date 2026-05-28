#!/usr/bin/env python3
"""Train decoder (latent -> IMS spectra) using precomputed latents and matching spectra.

Usage: python3 scripts/train_decoder.py --latents results/autoencoder_train_latent_1.npy \
    --train-feather Data/clean_split_1_of_5.feather --out models/decoder_split_1.pth
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def build_generator(latent_dim=512, output_dim=1676, n_layers=9):
    layer_sizes = list(np.linspace(latent_dim, output_dim, n_layers + 1).astype(int))
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < n_layers - 1:
            layers.append(nn.LeakyReLU(inplace=True))
    return nn.Sequential(*layers)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--latents', required=True)
    p.add_argument('--train-feather', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--latent-dim', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=9)
    return p.parse_args()


def load_feature_matrix(feather_path):
    df = pd.read_feather(feather_path)
    # drop meta columns
    drop = [c for c in ['Unnamed: 0', 'index', 'Label', 'TemperatureKelvin', 'temp_K', 'PressureBar'] if c in df.columns]
    df = df.drop(columns=drop, errors='ignore')
    # if class one-hot columns exist, drop them
    # assume feature columns are first (total_cols - 8)
    return df.iloc[:, :1676].values.astype('float32')


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    latents = np.load(args.latents)
    X = load_feature_matrix(args.train_feather)

    if len(latents) != len(X):
        raise SystemExit(f"Mismatch: {len(latents)} latents vs {len(X)} spectra in {args.train_feather}")

    # dataset
    X_t = torch.from_numpy(X)
    z_t = torch.from_numpy(latents.astype('float32'))

    dataset = TensorDataset(z_t, X_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_generator(args.latent_dim, 1676, args.n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for z_batch, x_batch in loader:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)
            pred = model(z_batch)
            loss = criterion(pred, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * z_batch.size(0)
        avg = total / len(dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: loss={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({'epoch': epoch, 'generator_state_dict': model.state_dict(), 'loss': avg}, args.out)

    print(f"Saved best decoder -> {args.out} (loss={best_loss:.6f})")


if __name__ == '__main__':
    main()
