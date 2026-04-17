#!/usr/bin/env python3
import os
import sys

import numpy as np
import pandas as pd
import torch


ROOT_DIR = "/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation"
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DATA_PATH = os.path.join(ROOT_DIR, "Data", "test_data.feather")
ENCODER_PATH = "/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ims_to_chemnet_encoder.pth"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "autoencoder_test_latent.npy")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    sys.path.insert(0, "/home/kjmetzler/ChemicalDataGeneration/models")
    model = torch.load(ENCODER_PATH, map_location=device, weights_only=False)
    model.eval()

    df = pd.read_feather(DATA_PATH)
    feature_cols = [c for c in df.columns if c.startswith("p_") or c.startswith("n_")]
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=device)

    batch_size = 4096
    latents = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            latents.append(model(x[i:i + batch_size]).cpu().numpy())

    test_latent = np.vstack(latents)
    np.save(OUTPUT_PATH, test_latent)
    print(f"Saved {OUTPUT_PATH} with shape {test_latent.shape}", flush=True)


if __name__ == "__main__":
    main()
