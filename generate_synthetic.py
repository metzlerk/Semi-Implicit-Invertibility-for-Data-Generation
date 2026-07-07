import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_path = "/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/train_data_with_conditions.feather"
encoder_path = "conditional_encoder.pt"
decoder_path = "conditional_decoder.pt"

df = pd.read_feather(train_path)

input_cols = [c for c in df.columns if c.startswith("p_") or c.startswith("n_") or c == "TemperatureKelvin"]
df = df.dropna(subset=input_cols)

X = df[input_cols].values.astype("float32")
temp_target = df["TemperatureKelvin"].values.astype("float32").reshape(-1, 1)

chem_cols = ["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]

# Keep original casing from the one-hot columns
df["chemical_name"] = df[chem_cols].idxmax(axis=1).str.strip()
chem_classes = df["chemical_name"].unique().tolist()
print("Classes:", chem_classes)

X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X_norm = (X - X_mean) / X_std

temp_mean = temp_target.mean()
temp_std = temp_target.std() + 1e-8
temp_norm = (temp_target - temp_mean) / temp_std

temp_idx = input_cols.index("TemperatureKelvin")
X_norm[:, temp_idx] = temp_norm[:, 0]

input_dim = X_norm.shape[1]
model1_dims = [input_dim, 1400, 1200, 1000, 800, 600, 513]
model2_dims = [513, 600, 800, 1000, 1200, 1400, input_dim]

encoder = MLP(model1_dims).to(device)
decoder = MLP(model2_dims).to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

encoder.eval()
decoder.eval()

with torch.no_grad():
    Z = encoder(torch.tensor(X_norm, dtype=torch.float32).to(device)).cpu().numpy()

plt.figure(figsize=(8, 6))
for chem in chem_classes:
    idx = np.where(df["chemical_name"].values == chem)[0]
    plt.scatter(Z[idx, 0], Z[idx, 1], s=5, alpha=0.5, label=chem)
plt.legend(markerscale=2, fontsize=8)
plt.xlabel("Latent dim 0")
plt.ylabel("Latent dim 1")
plt.tight_layout()
plt.savefig("latent_space_classes.png", dpi=200)
plt.close()
print("Saved latent_space_classes.png")

samples_per_class = 10000
temp_low = 301.0
temp_high = 310.0
batch_decode_size = 512

synthetic_rows = []

for chem in chem_classes:
    print(f"Generating synthetic samples for class: {chem}")

    idx = np.where(df["chemical_name"].values == chem)[0]
    Z_class = Z[idx]

    latent_std = np.std(Z_class[:, :512], axis=0)
    perturb_std = 0.1 * np.mean(latent_std)

    num_samples = samples_per_class
    base_indices = np.random.choice(len(Z_class), size=num_samples, replace=True)
    base_latent = Z_class[base_indices].copy()

    noise = np.random.normal(0, perturb_std, size=(num_samples, 512))
    base_latent[:, :512] += noise

    new_temps_phys = np.random.uniform(temp_low, temp_high, size=(num_samples, 1))
    new_temps_norm = (new_temps_phys - temp_mean) / temp_std
    base_latent[:, 512] = new_temps_norm[:, 0]

    decoded_batches = []
    with torch.no_grad():
        for start in range(0, num_samples, batch_decode_size):
            end = min(start + batch_decode_size, num_samples)
            batch_latent = torch.tensor(base_latent[start:end], dtype=torch.float32).to(device)
            decoded = decoder(batch_latent).cpu().numpy()
            decoded_batches.append(decoded)

    decoded_norm = np.vstack(decoded_batches)
    decoded_phys = decoded_norm * X_std + X_mean

    temp_decoded = decoded_phys[:, temp_idx]
    valid_mask = (
        (~np.isnan(decoded_phys).any(axis=1)) &
        (~np.isinf(decoded_phys).any(axis=1)) &
        (temp_decoded > 250) &
        (temp_decoded < 400)
    )

    decoded_phys = decoded_phys[valid_mask]
    new_temps_phys = new_temps_phys[valid_mask]

    print(f"Kept {decoded_phys.shape[0]} / {num_samples} samples for {chem}")

    for i in range(decoded_phys.shape[0]):
        row = {"chemical_name": chem, "TargetTemperatureKelvin": float(new_temps_phys[i, 0])}
        for j, col in enumerate(input_cols):
            if col == "TemperatureKelvin":
                row[col] = float(new_temps_phys[i, 0])
            else:
                row[col] = float(decoded_phys[i, j])
        synthetic_rows.append(row)

synthetic_df = pd.DataFrame(synthetic_rows)

# Use the same class names as chemical_name values
class_list = chem_classes
for cname in class_list:
    synthetic_df[cname] = (synthetic_df["chemical_name"] == cname).astype(int)

synthetic_df.to_feather("synthetic_high_temp.feather")
print("Saved synthetic_high_temp.feather")
