import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Load training data
# -----------------------------
train_path = "/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/train_data_with_conditions.feather"
chemnet_path = "/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/name_smiles_embedding_file.csv"

df = pd.read_feather(train_path)

# Drop junk columns
drop_cols = ["Unnamed: 0", "index", "PressureBar", "Label"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Select p_*, n_*, and TemperatureKelvin
input_cols = [c for c in df.columns if c.startswith("p_") or c.startswith("n_") or c == "TemperatureKelvin"]

# -----------------------------
# Remove rows with NaNs BEFORE anything else
# -----------------------------
print("Rows before NaN removal:", len(df))
df = df.dropna(subset=input_cols)
print("Rows after NaN removal:", len(df))

# Extract X and temperature
X = df[input_cols].values.astype("float32")
temp_target = df["TemperatureKelvin"].values.astype("float32").reshape(-1, 1)

# -----------------------------
# Extract chemical identity
# -----------------------------
chem_cols = ["DEB", "DEM", "DMMP", "DPM", "DtBP", "JP8", "MES", "TEPO"]
df["chemical_name"] = df[chem_cols].idxmax(axis=1)
df["chemical_name"] = df["chemical_name"].str.strip().str.upper()

# -----------------------------
# Load and parse ChemNet embeddings
# -----------------------------
chem = pd.read_csv(chemnet_path)

# Rename the unnamed index column
chem = chem.rename(columns={chem.columns[0]: "Abbrev"})

# Drop background row
chem = chem[chem["Abbrev"] != "BKG"]

# Normalize abbreviations
chem["Abbrev"] = chem["Abbrev"].astype(str).str.strip().str.upper()

# Parse embedding strings
chem["embedding"] = chem["embedding"].apply(ast.literal_eval)
chem["embedding"] = chem["embedding"].apply(lambda x: np.array(x, dtype="float32"))

# Set index to abbreviation
chem = chem.set_index("Abbrev")

print("ChemNet abbreviations:", chem.index.tolist())

# Align embeddings to training rows
chemnet_target = np.vstack([
    chem.loc[name, "embedding"]
    for name in df["chemical_name"]
]).astype("float32")

# -----------------------------
# Normalization & sanity checks
# -----------------------------
print("Before normalization:")
print("X NaNs:", np.isnan(X).sum(), "Infs:", np.isinf(X).sum())
print("ChemNet NaNs:", np.isnan(chemnet_target).sum(), "Infs:", np.isinf(chemnet_target).sum())
print("Temp NaNs:", np.isnan(temp_target).sum(), "Infs:", np.isinf(temp_target).sum())

assert np.isnan(X).sum() == 0 and np.isinf(X).sum() == 0, "X contains NaNs or Infs"
assert np.isnan(chemnet_target).sum() == 0 and np.isinf(chemnet_target).sum() == 0, "ChemNet contains NaNs or Infs"
assert np.isnan(temp_target).sum() == 0 and np.isinf(temp_target).sum() == 0, "Temp contains NaNs or Infs"

# Normalize X (feature-wise)
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True) + 1e-8
X = (X - X_mean) / X_std

# Normalize ChemNet embeddings (L2 per row)
norms = np.linalg.norm(chemnet_target, axis=1, keepdims=True) + 1e-8
chemnet_target = chemnet_target / norms

# Normalize temperature (global)
temp_mean = temp_target.mean()
temp_std = temp_target.std() + 1e-8
temp_target = (temp_target - temp_mean) / temp_std

print("After normalization:")
print("X max/min:", X.max(), X.min())
print("ChemNet max/min:", chemnet_target.max(), chemnet_target.min())
print("Temp max/min:", temp_target.max(), temp_target.min())

# -----------------------------
# Dataset
# -----------------------------
class MLPDataset(Dataset):
    def __init__(self, X, chemnet, temp):
        self.X = torch.tensor(X)
        self.chemnet = torch.tensor(chemnet)
        self.temp = torch.tensor(temp)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.chemnet[idx], self.temp[idx]

dataset = MLPDataset(X, chemnet_target, temp_target)
loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

input_dim = X.shape[1]

# -----------------------------
# Model Definitions
# -----------------------------
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

# Model 1: Input → 513
model1_dims = [input_dim, 1400, 1200, 1000, 800, 600, 513]
model1 = MLP(model1_dims).to(device)

# Model 2: 513 → Input
model2_dims = [513, 600, 800, 1000, 1200, 1400, input_dim]
model2 = MLP(model2_dims).to(device)

# -----------------------------
# Optimizers & Loss
# -----------------------------
opt1 = torch.optim.Adam(model1.parameters(), lr=1e-5)
opt2 = torch.optim.Adam(model2.parameters(), lr=1e-5)

mse = nn.MSELoss()

# -----------------------------
# Training Loop
# -----------------------------
epochs = 2000
patience = 20
best_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(epochs):
    total_loss1 = 0.0
    total_loss2 = 0.0

    for batch_x, batch_chem, batch_temp in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_chem = batch_chem.to(device, non_blocking=True)
        batch_temp = batch_temp.to(device, non_blocking=True)

        # -------------------------
        # Train Model 1
        # -------------------------
        opt1.zero_grad()
        out1 = model1(batch_x)

        chem_pred = out1[:, :512]
        temp_pred = out1[:, 512].unsqueeze(1)

        loss1 = mse(chem_pred, batch_chem) + mse(temp_pred, batch_temp)

        if torch.isnan(loss1):
            raise RuntimeError("NaN detected in Model1 loss")

        loss1.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
        opt1.step()

        total_loss1 += loss1.item()

        # -------------------------
        # Train Model 2
        # -------------------------
        opt2.zero_grad()
        out2 = model2(out1.detach())

        loss2 = mse(out2, batch_x)

        if torch.isnan(loss2):
            raise RuntimeError("NaN detected in Model2 loss")

        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt2.step()

        total_loss2 += loss2.item()

    print(f"Epoch {epoch+1}/{epochs} | Model1 Loss: {total_loss1:.4f} | Model2 Loss: {total_loss2:.4f}")
    # Early stopping check
    if total_loss1 < best_loss:
        best_loss = total_loss1
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break


# -----------------------------
# Save Models
# -----------------------------
torch.save(model1.cpu().state_dict(), "conditional_encoder.pt")
torch.save(model2.cpu().state_dict(), "conditional_decoder.pt")

print("Models saved as conditional_encoder.pt and conditional_decoder.pt")
