#!/usr/bin/env python3
"""
Generate autoencoder latent .npy files using the pretrained encoder
from the ChemicalDataGeneration repo.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build autoencoder latent .npy files.")
    parser.add_argument(
        "--chemgen-models-path",
        default="/home/kjmetzler/ChemicalDataGeneration/models",
        help="Path to ChemicalDataGeneration/models for importing functions.",
    )
    parser.add_argument(
        "--encoder-path",
        default=(
            "/home/kjmetzler/ChemicalDataGeneration/models/trained_models/spectrum/"
            "new_hypertuning_results_20251009_100119/baseline_exact_encoder.pth"
        ),
        help="Path to pretrained encoder .pth file.",
    )
    parser.add_argument(
        "--train-data",
        default="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/train_data.feather",
        help="Path to training spectra feather file.",
    )
    parser.add_argument(
        "--test-data",
        default="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/test_data.feather",
        help="Path to test spectra feather file.",
    )
    parser.add_argument(
        "--embedding-file",
        default="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/name_smiles_embedding_file.csv",
        help="Path to ChemNet embedding CSV.",
    )
    parser.add_argument(
        "--output-train",
        default="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/autoencoder_train_latent.npy",
        help="Output path for train latents .npy.",
    )
    parser.add_argument(
        "--output-test",
        default="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/autoencoder_test_latent.npy",
        help="Output path for test latents .npy.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding prediction.",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=2,
        help="Start index for spectral columns.",
    )
    parser.add_argument(
        "--stop-idx",
        type=int,
        default=-9,
        help="Stop index for spectral columns.",
    )
    parser.add_argument(
        "--use-column-slice",
        action="store_true",
        help="Use start/stop index slicing instead of auto-detected feature columns.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save embedding prediction CSVs alongside .npy files.",
    )
    return parser.parse_args()

CLASS_NAMES = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']


class BiasOnlyLayer(nn.Module):
    def __init__(self, bias_init):
        super().__init__()
        self.bias = nn.Parameter(bias_init.clone().detach())

    def forward(self, x):
        return x + self.bias


class BaselineEncoder(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.background_layer = BiasOnlyLayer(torch.zeros(layer_sizes[0]))
        layers = []
        for idx in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
        self.layers = nn.ModuleList(layers)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.background_layer(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.activation(x)
        return x


class EncoderBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["input_dim"]
        hidden_dims = config["hidden_dims"]
        output_dim = config["output_dim"]
        layer_dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(layer_dims[i], layer_dims[i + 1]) for i in range(len(layer_dims) - 1)]
        )
        self.batch_norms = nn.ModuleList()
        if config.get("use_batch_norm"):
            for dim in layer_dims[1:]:
                channels = 32
                if dim % channels != 0:
                    raise ValueError(
                        f"Cannot apply batch norm with {channels} channels to dimension {dim}."
                    )
                self.batch_norms.append(nn.BatchNorm1d(channels))
        self.activations = config.get("activations", ["tanh"] * len(hidden_dims))
        self.dropout_rates = config.get("dropout_rates", [0.0] * len(hidden_dims))
        self.use_residual = config.get("use_residual", False)

    def _apply_activation(self, x, idx):
        act = self.activations[idx] if idx < len(self.activations) else "tanh"
        if act == "relu":
            return torch.relu(x)
        if act == "tanh":
            return torch.tanh(x)
        if act == "leaky_relu":
            return torch.nn.functional.leaky_relu(x)
        return x

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self._apply_activation(x, idx)
                if self.batch_norms:
                    batch = x.shape[0]
                    x = x.view(batch, 32, -1)
                    x = self.batch_norms[idx](x)
                    x = x.view(batch, -1)
                if idx < len(self.dropout_rates):
                    drop = self.dropout_rates[idx]
                    if drop > 0:
                        x = torch.nn.functional.dropout(x, p=drop, training=self.training)
                if self.use_residual and residual.shape == x.shape:
                    x = x + residual
        return x


class ConfigurableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderBackbone(config)

    def forward(self, x):
        return self.encoder(x)


def ensure_index_column(df):
    if "index" not in df.columns:
        df = df.copy()
        df.insert(0, "index", np.arange(len(df)))
    return df


def normalize_labels(df):
    if "Label" not in df.columns:
        raise ValueError("Expected Label column in dataset.")
    labels = df["Label"]
    if pd.api.types.is_numeric_dtype(labels):
        label_indices = labels.astype(int).to_numpy()
        if np.any((label_indices < 0) | (label_indices >= len(CLASS_NAMES))):
            raise ValueError("Numeric labels are out of range for CLASS_NAMES.")
        df = df.copy()
        df["Label"] = [CLASS_NAMES[idx] for idx in label_indices]
    elif not set(labels.unique()).issubset(set(CLASS_NAMES)):
        raise ValueError("Label column contains unexpected class names.")
    return df


def ensure_class_columns(df):
    df = normalize_labels(df)
    class_cols = [name for name in CLASS_NAMES if name in df.columns]
    if len(class_cols) != len(CLASS_NAMES):
        one_hot = pd.get_dummies(pd.Categorical(df["Label"], categories=CLASS_NAMES)).astype(int)
        for name in CLASS_NAMES:
            df[name] = one_hot[name].to_numpy()
    return df


def load_dataset(path):
    df = pd.read_feather(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = ensure_index_column(df)
    df = ensure_class_columns(df)
    return df


def build_dataset(df, embedding_df, device, start_idx, stop_idx, use_column_slice):
    if use_column_slice:
        embeddings_tensor, spectra_tensor, chem_encodings_tensor, spectra_indices_tensor = (
            cfunc.create_dataset_tensors(
                df,
                embedding_df,
                device,
                start_idx=start_idx,
                stop_idx=stop_idx,
            )
        )
        return TensorDataset(
            spectra_tensor,
            chem_encodings_tensor,
            embeddings_tensor,
            spectra_indices_tensor,
        )

    class_cols = [name for name in CLASS_NAMES if name in df.columns]
    exclude = set(["index", "Label"]) | set(CLASS_NAMES)
    feature_cols = [col for col in df.columns if col not in exclude]
    if len(feature_cols) != 1676:
        raise ValueError(
            f"Expected 1676 feature columns, found {len(feature_cols)}. "
            "Use --use-column-slice if your dataset uses a different layout."
        )
    embeddings_tensor = torch.Tensor(
        [embedding_df["Embedding Floats"][chem_name] for chem_name in df["Label"]]
    ).to(device)
    spectra_tensor = torch.Tensor(df[feature_cols].values).to(device)
    chem_encodings_tensor = torch.Tensor(df[class_cols].values).to(device)
    spectra_indices_tensor = torch.Tensor(df["index"].to_numpy()).to(device)
    return TensorDataset(
        spectra_tensor,
        chem_encodings_tensor,
        embeddings_tensor,
        spectra_indices_tensor,
    )


def predict_embeddings(df, embedding_df, model, device, batch_size, start_idx, stop_idx, use_column_slice):
    chem_names = CLASS_NAMES
    dataset = build_dataset(df, embedding_df, device, start_idx, stop_idx, use_column_slice)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.MSELoss()
    preds, name_encodings, avg_loss, indices = cfunc.predict_embeddings(
        loader, model, device, criterion
    )
    preds_df = cfunc.format_preds_df(indices, preds, name_encodings, chem_names)
    preds_df = preds_df.sort_values("index").reset_index(drop=True)
    embedding_df = preds_df.drop(columns=["index"] + chem_names)
    return embedding_df.to_numpy(), preds_df, avg_loss


if __name__ == "__main__":
    args = parse_args()

    sys.path.append(args.chemgen_models_path)
    import functions as cfunc  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_df = cfunc.format_embedding_df(args.embedding_file)

    print("Loading encoder model...")
    checkpoint = torch.load(args.encoder_path, weights_only=False, map_location=device)
    if isinstance(checkpoint, nn.Module):
        encoder = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        encoder = ConfigurableEncoder(checkpoint["config"])
        encoder.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        layer_sizes = [1676, 1548, 1420, 1292, 1164, 1036, 908, 780, 652, 512]
        encoder = BaselineEncoder(layer_sizes)
        encoder.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported encoder checkpoint format: {type(checkpoint)}")

    encoder.to(device)
    encoder.eval()

    print("Loading train data...")
    train_df = load_dataset(args.train_data)
    print("Generating train embeddings...")
    train_latents, train_preds_df, train_loss = predict_embeddings(
        train_df,
        embedding_df,
        encoder,
        device,
        args.batch_size,
        args.start_idx,
        args.stop_idx,
        args.use_column_slice,
    )
    np.save(args.output_train, train_latents)
    print(f"Saved train latents: {args.output_train} (loss={train_loss:.6f})")
    if args.save_csv:
        train_csv = os.path.splitext(args.output_train)[0] + "_preds.csv"
        train_preds_df.to_csv(train_csv, index=False)
        print(f"Saved train predictions: {train_csv}")

    print("Loading test data...")
    test_df = load_dataset(args.test_data)
    print("Generating test embeddings...")
    test_latents, test_preds_df, test_loss = predict_embeddings(
        test_df,
        embedding_df,
        encoder,
        device,
        args.batch_size,
        args.start_idx,
        args.stop_idx,
        args.use_column_slice,
    )
    np.save(args.output_test, test_latents)
    print(f"Saved test latents: {args.output_test} (loss={test_loss:.6f})")
    if args.save_csv:
        test_csv = os.path.splitext(args.output_test)[0] + "_preds.csv"
        test_preds_df.to_csv(test_csv, index=False)
        print(f"Saved test predictions: {test_csv}")
