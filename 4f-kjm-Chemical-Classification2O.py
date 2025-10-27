import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from iterativenn.iterativenn.src.iterativenn import utils
from iterativenn.iterativenn.src.iterativenn.nn_modules.MaskedLinear import MaskedLinear
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#%%
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# Load data
train = pd.read_feather('/home/kjmetzler/train_data_subset.feather')
test = pd.read_feather('/home/kjmetzler/test_data_subset.feather')

# Function to sample instances per class
def sample_per_class(df, label_col, n_samples=50):
    return df.groupby(label_col).apply(lambda x: x.sample(n=min(len(x), n_samples), random_state=42)).reset_index(drop=True)

# Sample data
df = sample_per_class(train, 'Label', 1000)
testing = test

# One-hot encoding
def one_hot_encode(df, label_col):
    one_hot_labels = pd.get_dummies(df[label_col], prefix=label_col).astype(int)
    df = df.drop(label_col, axis=1)
    return df, one_hot_labels

df, one_hot_labels = one_hot_encode(df, 'Label')
testing, _ = one_hot_encode(testing, 'Label')

label_size = one_hot_labels.shape[1]
data_size = df.shape[1] - label_size

# Add noise function
def add_noise(dataframe, n=1):
    labels = dataframe.iloc[:, -label_size:]
    chemicals = dataframe.iloc[:, :-label_size]
    all_blurred_chemicals = []
    blur_counts = []

    for _ in range(n):
        blurred_chemicals = chemicals.rolling(window=3, min_periods=1, axis=1).mean().values
        blurred_chemicals = np.clip(blurred_chemicals, 0, None)
        max_entry = np.max(blurred_chemicals)
        if max_entry > 0:
            blurred_chemicals = (blurred_chemicals / max_entry) * 100
        all_blurred_chemicals.append(blurred_chemicals)
        blur_counts.extend([_ + 1] * len(chemicals))

    blurred_chemicals_combined = np.vstack(all_blurred_chemicals)
    blurred_labels_combined = np.vstack([labels.values] * n)
    blur_counts_column = np.array(blur_counts).reshape(-1, 1)
    
    return pd.concat([pd.DataFrame(blurred_chemicals_combined), pd.DataFrame(blurred_labels_combined), pd.DataFrame(blur_counts_column, columns=['Blur_Count'])], axis=1, ignore_index=True)

# Convert dataframe to tensor
def df_to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)

# Initialize MaskedLinear layer
def initialize_masked_linear():
    row_sizes = [2*data_size, 2*label_size]
    col_sizes = [2*df.shape[1]]
    block_types = [['R=0.1'], ['R=0.1']]
    initialization_types = [[1], ['G']]
    trainable = [[1], [1]]
    chem_ml = MaskedLinear(3372, 3372, bias=True)
    return chem_ml.from_description(row_sizes=row_sizes, col_sizes=col_sizes, block_types=block_types, initialization_types=initialization_types, trainable=trainable)

chem_MaskLin = initialize_masked_linear()
chem_INN = torch.nn.Sequential(chem_MaskLin, nn.LeakyReLU(0.1)).to(device)

# Prepare data for training
n = 10

# Add random column entries before the last 8 entries to pad the length of the samples
def pad_with_random_entries(df, pad_size):
    padding = np.random.rand(df.shape[0], pad_size)
    padded_df = np.hstack((df.values[:, :-label_size], padding, df.values[:, -label_size:]))
    return pd.DataFrame(padded_df)

# Calculate the required padding size to make the total number of columns x_size*2
x_size = df.shape[1]
required_padding_size = x_size * 2 - df.shape[1] - 1

# Calculate the required padding size to make the total number of columns 3372
required_padding_size = 3372 - df.shape[1] - 1

# Pad the dataframe
df_padded = pad_with_random_entries(df, required_padding_size)

# Convert to tensor
x_start_tensor = df_to_tensor(add_noise(df_padded, n)).to(device)
print(x_start_tensor.shape)

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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

# Training setup
criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion2 = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(chem_INN.parameters(), lr=0.0001)
max_epochs = 2000
loss_graph = []

# Training loop
for epoch in range(max_epochs):
    for batch_idx, (start, target) in enumerate(train_loader):
        optimizer.zero_grad()
        start, target = start.to(device), target.to(device)
        set = start
        loss = 0
        for i in range(5):
            set = chem_INN(set)
            soft_guess = set[:, -label_size:]
            mass_spec = set[:, :data_size]
            loss += criterion(soft_guess, target[:, -label_size:]) * 20 * (i + 1) ** 2
            loss += criterion2(mass_spec, target[:, :data_size])
        loss.backward()
        optimizer.step()
    loss_graph.append(loss.item())
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}')
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

# Create test dataset
def create_test_dataset(dataset, add_noise, std=0.1):
    num_features = data_size
    num_labels = label_size
    mass_specs = dataset.iloc[:, :num_features]
    labels = dataset.iloc[:, num_features:]
    test_samples = []
    target_samples = []
    combination_limit = 100

    for i in range(len(dataset)):
        combinations_made = 0
        for j in range(i + 1, len(dataset)):
            label_i = labels.iloc[i]
            label_j = labels.iloc[j]
            if label_i.equals(label_j):
                noisy_mass_spec_i = add_noise(mass_specs.iloc[i].to_frame().T, 1).values
                noisy_mass_spec_j = add_noise(mass_specs.iloc[j].to_frame().T, 1).values
                avg_noisy_mass_spec = (noisy_mass_spec_i[:data_size] + noisy_mass_spec_j[:data_size]) / 2
                avg_mass_spec = (mass_specs.iloc[i][:data_size] + mass_specs.iloc[j][:data_size]) / 2
                duplicated_labels = np.tile(label_i.values, (1, 1))
                test_samples.append(np.concatenate([avg_noisy_mass_spec[:data_size], duplicated_labels], axis=1))
                target_samples.append(np.concatenate([avg_mass_spec.values[:data_size], label_i.values]))
                combinations_made += 1
                if combinations_made >= combination_limit:
                    break

    test_samples_flat = np.vstack(test_samples)
    target_samples_repeated = np.repeat(target_samples, repeats=1, axis=0)
    return pd.DataFrame(test_samples_flat), pd.DataFrame(target_samples_repeated)

# Example usage
sampled_df = sample_per_class(df, 'Label', 50)
X_train = sampled_df.iloc[:, :data_size].values
y_train = sampled_df.iloc[:, -label_size:].values
X_test = testing.iloc[:, :data_size].values
y_test = testing.iloc[:, -label_size:].values
train_data = np.hstack((X_train, y_train))
num_features = X_train.shape[1]
columns = [f"feature_{i}" for i in range(num_features)] + [f"class_{i}" for i in range(y_train.shape[1])]
train_data_df = pd.DataFrame(train_data, columns=columns)
test_dataset, test_target = create_test_dataset(train_data_df, add_noise, std=0.1)
padded_test_dataset = pad_with_random_entries(test_dataset, required_padding_size)
test_dataset_tensor = torch.tensor(padded_test_dataset.values, dtype=torch.float).to(device)
test_target_tensor = torch.tensor(test_target.values, dtype=torch.float).to(device)

# Use the generated test dataset with your network chem_INN()
output = chem_INN(chem_INN(chem_INN(chem_INN(chem_INN(test_dataset_tensor[:, :2*df.shape[1]])))))
output_label = output[:, -label_size:]
target_label = test_target_tensor[:, -label_size:]
output_labels = torch.argmax(output_label, dim=1).cpu().numpy()
target_labels = torch.argmax(target_label, dim=1).cpu().numpy()
conf_matrix = confusion_matrix(target_labels, output_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Random Forest Classifier
def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# MLP Classifier
def train_and_evaluate_mlp(X_train, y_train, X_test, y_test):
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(1000, 500, 250, 100), activation='relu', solver='adam', learning_rate='adaptive', max_iter=10000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Run tests
def run_tests(real_data, synthetic_data, test_data, num_points_per_class, ratios, classifier_fn):
    X_real, y_real = real_data.iloc[:, :data_size].values, np.argmax(real_data.iloc[:, -label_size:].values, axis=1)
    X_synthetic, y_synthetic = synthetic_data.iloc[:, :data_size].values, np.argmax(synthetic_data.iloc[:, -label_size:].values, axis=1)
    X_test, y_test = test_data.iloc[:, :data_size].values, np.argmax(test_data.iloc[:, -label_size:].values, axis=1)
    results = []

    for ratio in ratios:
        for num_points in num_points_per_class:
            X_train = []
            y_train = []
            for class_label in np.unique(y_real):
                real_indices = np.where(y_real == class_label)[0][:int(num_points * ratio + 0.5)]
                synthetic_indices = np.where(y_synthetic == class_label)[0][:int(num_points * (1 - ratio) + 0.5)]
                if len(real_indices) > 0:
                    X_train.append(X_real[real_indices])
                    y_train.append(y_real[real_indices])
                if len(synthetic_indices) > 0:
                    X_train.append(X_synthetic[synthetic_indices])
                    y_train.append(y_synthetic[synthetic_indices])
            if len(X_train) > 0 and len(y_train) > 0:
                X_train = np.vstack(X_train)
                y_train = np.hstack(y_train)
                accuracy = classifier_fn(X_train, y_train, X_test, y_test)
                results.append((num_points, ratio, accuracy, len(X_train)))
    return results

# Define the number of datapoints per class to test and the ratios of real to synthetic data
num_points_per_class = np.arange(1, 51)
ratios = np.linspace(0, 1, 11)

# Run the tests for Random Forest with Cate's Synthetic Data
results_rf = run_tests(train, train_data, testing, num_points_per_class, ratios, train_and_evaluate_rf)
num_points_rf, ratios_rf, accuracies_rf, num_datapoints_rf = zip(*results_rf)

# Plot the results for Random Forest with Cate's Synthetic Data
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ratios_rf))))
for i, ratio in enumerate(np.unique(ratios_rf)):
    mask = np.array(ratios_rf) == ratio
    plt.plot(np.array(num_points_rf)[mask], np.array(accuracies_rf)[mask], marker='o', label=f'Ratio {ratio:.1f}', color=colors[i])
plt.xlabel('Total Number of Datapoints per Class')
plt.ylabel('Accuracy')
plt.title('Random Forest: Accuracy vs. Number of Datapoints per Class')
plt.legend()
plt.grid(True)
plt.savefig('/home/kjmetzler/random_forest_accuracy_Cate.png')
plt.show()

# Run the tests for MLP with Cate's Synthetic Data
results_mlp = run_tests(train, train_data, testing, num_points_per_class, ratios, train_and_evaluate_mlp)
num_points_mlp, ratios_mlp, accuracies_mlp, num_datapoints_mlp = zip(*results_mlp)

# Plot the results for MLP with Cate's Synthetic Data
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ratios_mlp))))
for i, ratio in enumerate(np.unique(ratios_mlp)):
    mask = np.array(ratios_mlp) == ratio
    plt.plot(np.array(num_points_mlp)[mask], np.array(accuracies_mlp)[mask], marker='o', label=f'Ratio {ratio:.1f}', color=colors[i])
plt.xlabel('Total Number of Datapoints per Class')
plt.ylabel('Accuracy')
plt.title('MLP: Accuracy vs. Number of Datapoints per Class')
plt.legend()
plt.grid(True)
plt.savefig('/home/kjmetzler/mlp_accuracy_Cate.png')
plt.show()

# Run the tests for Random Forest with Kevin's Synthetic Data
results_rf_output = run_tests(train, pd.DataFrame(output.cpu().numpy()), testing, num_points_per_class, ratios, train_and_evaluate_rf)
num_points_rf_output, ratios_rf_output, accuracies_rf_output, num_datapoints_rf_output = zip(*results_rf_output)

# Plot the results for Random Forest with Kevin's Synthetic Data
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ratios_rf_output))))
for i, ratio in enumerate(np.unique(ratios_rf_output)):
    mask = np.array(ratios_rf_output) == ratio
    plt.plot(np.array(num_points_rf_output)[mask], np.array(accuracies_rf_output)[mask], marker='o', label=f'Ratio {ratio:.1f}', color=colors[i])
plt.xlabel('Total Number of Datapoints per Class')
plt.ylabel('Accuracy')
plt.title('Random Forest with Synthetic Data from Output: Accuracy vs. Number of Datapoints per Class')
plt.legend()
plt.grid(True)
plt.savefig('/home/kjmetzler/random_forest_accuracy_Kevin.png')
plt.show()

# Run the tests for MLP with Kevin's Synthetic Data
results_mlp_output = run_tests(train, pd.DataFrame(output.cpu().numpy()), testing, num_points_per_class, ratios, train_and_evaluate_mlp)
num_points_mlp_output, ratios_mlp_output, accuracies_mlp_output, num_datapoints_mlp_output = zip(*results_mlp_output)

# Plot the results for MLP with Kevin's Synthetic Data
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ratios_mlp_output))))
for i, ratio in enumerate(np.unique(ratios_mlp_output)):
    mask = np.array(ratios_mlp_output) == ratio
    plt.plot(np.array(num_points_mlp_output)[mask], np.array(accuracies_mlp_output)[mask], marker='o', label=f'Ratio {ratio:.1f}', color=colors[i])
plt.xlabel('Total Number of Datapoints per Class')
plt.ylabel('Accuracy')
plt.title('MLP with Synthetic Data from Output: Accuracy vs. Number of Datapoints per Class')
plt.legend()
plt.grid(True)
plt.savefig('/home/kjmetzler/mlp_accuracy_Kevin.png')
plt.show()



