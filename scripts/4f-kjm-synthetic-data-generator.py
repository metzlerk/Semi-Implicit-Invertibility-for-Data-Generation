import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
from iterativenn.iterativenn.src.iterativenn.nn_modules.MaskedLinear import MaskedLinear
import matplotlib.pyplot as plt
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start_time = time.time()

def elapsed_time():
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

print(f"{elapsed_time()}: Loading training data", flush=True)
train = pd.read_feather('/home/kjmetzler/noisy_data.feather')
print(f"{elapsed_time()}: Loading testing data", flush=True)
testing = pd.read_feather('/home/kjmetzler/test_data_subset.feather')

# One-hot encoding
def one_hot_encode(df, label_col):
    one_hot_labels = pd.get_dummies(df[label_col], prefix=label_col).astype(int)
    df.drop(label_col, axis=1, inplace=True)
    return df, one_hot_labels

print(f"{elapsed_time()}: One-hot encoding testing data", flush=True)
testing, one_hot_labels = one_hot_encode(testing, 'Label')
del testing

label_size = one_hot_labels.shape[1]
data_size = train.shape[1] - label_size
train_shape = data_size + label_size - 1

# Create synthetic dataset
def create_synthetic_data(dataset):
    num_features = dataset.shape[1] - 8
    
    mass_specs = dataset.iloc[:, :num_features]
    labels = dataset.iloc[:, num_features:]
    test_samples = []
    target_samples = []
    
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"{elapsed_time()}: Processing row {i} of {len(dataset)}", end='\r', flush=True)
        label_i = labels.iloc[i]
        same_class_indices = labels[labels.eq(label_i).all(axis=1)].index.tolist()
        same_class_indices.remove(i)
        
        if same_class_indices:
            j = np.random.choice(same_class_indices)
            label_j = labels.iloc[j]
            avg_mass_spec = (mass_specs.iloc[i][:data_size] + mass_specs.iloc[j][:data_size]) / 2
            duplicated_labels = np.tile(label_i.values, (1, 1))
            test_samples.append(np.concatenate([avg_mass_spec[:data_size], duplicated_labels.flatten()]))
            target_samples.append(np.concatenate([avg_mass_spec.values[:data_size], label_i.values]))

    test_samples_flat = np.vstack(test_samples)
    target_samples_repeated = np.vstack(target_samples)
    return pd.DataFrame(test_samples_flat), pd.DataFrame(target_samples_repeated)

print(f"{elapsed_time()}: Creating synthetic data", flush=True)
sampled_df = train
del train
X_train = sampled_df.iloc[:, :data_size].values
y_train = sampled_df.iloc[:, -label_size:].values
del sampled_df
train_data = np.hstack((X_train, y_train))
num_features = X_train.shape[1]
del X_train, y_train
columns = [f"feature_{i}" for i in range(num_features)] + [f"class_{i}" for i in range(label_size)]
train_data_df = pd.DataFrame(train_data, columns=columns)
del train_data
test_dataset, test_target = create_synthetic_data(train_data_df)
del train_data_df

print(f"{elapsed_time()}: Converting datasets to tensors", flush=True)
test_dataset_tensor = torch.tensor(test_dataset.values, dtype=torch.float).to(device)
del test_dataset
test_target_tensor = torch.tensor(test_target.values, dtype=torch.float).to(device)
del test_target

# Use the generated test dataset with your network chem_INN()
# Load the trained model
# Initialize MaskedLinear layer
def initialize_masked_linear():
    row_sizes = [data_size-1, label_size, 1]
    col_sizes = [data_size+label_size]
    block_types = [['R=0.1'], ['R=0.1'], ['W']]
    initialization_types = [[1], ['G'], ['G']]
    trainable = [[1], [1], [1]]
    chem_ml = MaskedLinear(train_shape+1, train_shape+1, bias=True)
    return chem_ml.from_description(row_sizes=row_sizes, col_sizes=col_sizes, block_types=block_types, initialization_types=initialization_types, trainable=trainable)

print(f"{elapsed_time()}: Initializing MaskedLinear layer", flush=True)
chem_MaskLin = initialize_masked_linear()
chem_INN = torch.nn.Sequential(chem_MaskLin, torch.nn.LeakyReLU(0.1)).to(device)

print(f"{elapsed_time()}: Loading trained model", flush=True)
model = chem_INN
model.load_state_dict(torch.load('/home/kjmetzler/trained_model.pth'))
model.eval()

print(f"{elapsed_time()}: Generating output using the trained model", flush=True)
output = test_dataset_tensor[:, :]
del test_dataset_tensor
with torch.no_grad():
    for _ in range(5):
        output = model(output)

print(f"{elapsed_time()}: Processing final output", flush=True)
output_label = output[:, -label_size:]
target_label = test_target_tensor[:, -label_size:]
del test_target_tensor
output_labels = torch.argmax(output_label, dim=1).cpu().numpy()
del output_label
target_labels = torch.argmax(target_label, dim=1).cpu().numpy()
del target_label
conf_matrix = confusion_matrix(target_labels, output_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print(f"{elapsed_time()}: Saving generated synthetic data", flush=True)
output_df = pd.DataFrame(output.cpu().numpy())
output_labels_df = pd.DataFrame(output_labels, columns=['Label'])
output_df = pd.concat([output_df, output_labels_df], axis=1)
del output
output_df.to_csv('/home/kjmetzler/generated_synthetic_data.csv', index=False)

# Clean up to free memory
print(f"{elapsed_time()}: Cleaning up and freeing memory", flush=True)
torch.cuda.empty_cache()