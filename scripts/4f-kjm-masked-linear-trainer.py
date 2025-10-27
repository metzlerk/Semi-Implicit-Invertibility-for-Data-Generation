import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from iterativenn.iterativenn.src.iterativenn.nn_modules.MaskedLinear import MaskedLinear

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train = pd.read_feather('/home/kjmetzler/train_data_subset.feather')

# Sample data
df = train

# One-hot encoding
def one_hot_encode(df, label_col):
    one_hot_labels = pd.get_dummies(df[label_col], prefix=label_col).astype(int)
    df = df.drop(label_col, axis=1)
    return df, one_hot_labels

df, one_hot_labels = one_hot_encode(df, 'Label')
# Drop the original 'Label' column and concatenate the one-hot encoded labels
df = df.drop('Unnamed: 0', axis=1)
df = df.drop("index", axis=1)

label_size = one_hot_labels.shape[1]
data_size = df.shape[1] - label_size

# Convert dataframe to tensor
def df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

# Initialize MaskedLinear layer
def initialize_masked_linear():
    row_sizes = [data_size, label_size, 1]
    col_sizes = [data_size+1+label_size]
    block_types = [['R=0.1'], ['R=0.1'], ['W']]
    initialization_types = [[1], ['G'], ['G']]
    trainable = [[1], [1], [1]]
    chem_ml = MaskedLinear(df.shape[1]+1, df.shape[1]+1, bias=True)
    return chem_ml.from_description(row_sizes=row_sizes, col_sizes=col_sizes, block_types=block_types, initialization_types=initialization_types, trainable=trainable)

chem_MaskLin = initialize_masked_linear()
chem_INN = torch.nn.Sequential(chem_MaskLin, nn.LeakyReLU(0.1)).to(device)

# Prepare data for training
n = 10

# Convert to tensor
noisy_data = pd.read_feather('/home/kjmetzler/noisy_data.feather')
x_start_tensor = df_to_tensor(noisy_data).to(device)

df_repeated = pd.concat([df] * n, ignore_index=True)
x_target_tensor = df_to_tensor(df_repeated).to(device)

# Dataset class
class Data(torch.utils.data.Dataset):
    def __init__(self, x_start, x_target):
        self.x_start = x_start
        self.x_target = x_target
    def __len__(self):
        return len(self.x_start)
    def __getitem__(self, idx):
        return self.x_start[idx], self.x_target[idx]

train_data = Data(x_start_tensor, x_target_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

# Training setup
criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion2 = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(chem_INN.parameters(), lr=0.0001)
max_epochs = 2000
loss_graph = []

# Training loop
for epoch in range(max_epochs):
    
    chem_INN.train()  # Set the model to training mode
    for batch_idx, (start, target) in enumerate(train_loader):
        optimizer.zero_grad()
        start, target = start.to(device, non_blocking=True), target.to(device, non_blocking=True)
        set = start
        loss = 0
        for i in range(5):
            set = chem_INN(set)
            soft_guess = set[:, -(label_size+1):-1]
            mass_spec = set[:, :data_size]
            count = set[:, -1]
            loss += criterion(soft_guess, target[:, -label_size:]) * 20 * (i + 1) ** 2
            loss += criterion2(mass_spec, target[:, :data_size])
            loss += criterion2(count, start[:, -1])
        loss.backward()
        optimizer.step()
    loss_graph.append(loss.item())
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
        torch.cuda.empty_cache()

# Save loss graph
plt.figure()
plt.plot(np.arange(0, len(loss_graph)), loss_graph)
plt.semilogy()
plt.savefig('/home/kjmetzler/loss_graph.png')
plt.close()
torch.cuda.empty_cache()

# Confusion Matrix
pred_output = chem_INN(chem_INN(chem_INN(chem_INN(chem_INN(x_start_tensor)))))
pred_label = pred_output[:, -label_size:]
target_label = x_target_tensor[:, -label_size:]
pred_classes = torch.argmax(pred_label, dim=1).cpu().numpy()
target_classes = torch.argmax(target_label, dim=1).cpu().numpy()
conf_matrix = confusion_matrix(target_classes, pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Save the trained model
model_path = '/home/kjmetzler/trained_model.pth'
torch.save(chem_INN.state_dict(), model_path)
print(f'Model saved to {model_path}')
