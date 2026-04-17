import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from datetime import datetime
import time


start_time = time.time()

def elapsed_time():
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

def print_with_timestamp(message):
    print(f"{elapsed_time()}: {message}", flush=True)

print_with_timestamp("Loading data...")
testing = pd.read_feather('/home/kjmetzler/test_data_subset.feather')
real_data = pd.read_feather('/home/kjmetzler/train_data_subset.feather')

# Load synthetic datasets from numpy files
print_with_timestamp("Loading synthetic datasets...")
class_names = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

def load_synthetic_data(spectra_path, labels_path):
    spectra = np.load(spectra_path)
    labels = np.load(labels_path)
    
    # Create DataFrame with features
    df = pd.DataFrame(spectra)
    
    # One-hot encode labels
    one_hot = np.zeros((len(labels), len(class_names)))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    one_hot_df = pd.DataFrame(one_hot, columns=class_names)
    
    # Concatenate features and one-hot labels
    return pd.concat([df, one_hot_df], axis=1)

std1_data = load_synthetic_data(
    '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.0.npy',
    '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.0.npy'
)
std15_data = load_synthetic_data(
    '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.5.npy',
    '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.5.npy'
)
std2_data = load_synthetic_data(
    '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std2.0.npy',
    '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std2.0.npy'
)

testing = testing.drop(columns=['Unnamed: 0', 'index', 'Label'], axis=1)
real_data = real_data.drop(columns=['Unnamed: 0', 'index', 'Label'], axis=1)

label_size = len(class_names)
data_size = real_data.shape[1] - label_size

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

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
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

# Test all three synthetic datasets
synthetic_datasets = [
    (std1_data, 'std1.0', 'Synthetic Data (std=1.0)'),
    (std15_data, 'std1.5', 'Synthetic Data (std=1.5)'),
    (std2_data, 'std2.0', 'Synthetic Data (std=2.0)')
]

for syn_data, std_label, title_label in synthetic_datasets:
    # Random Forest
    print_with_timestamp(f"Running tests for Random Forest with {title_label}...")
    results_rf = run_tests(real_data, syn_data, testing, num_points_per_class, ratios, train_and_evaluate_rf)
    num_points_rf, ratios_rf, accuracies_rf, num_datapoints_rf = zip(*results_rf)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ratios_rf))))
    for i, ratio in enumerate(np.unique(ratios_rf)):
        mask = np.array(ratios_rf) == ratio
        plt.plot(np.array(num_points_rf)[mask], np.array(accuracies_rf)[mask], marker='o', label=f'Ratio {ratio:.1f}', color=colors[i])
    plt.xlabel('Total Number of Datapoints per Class')
    plt.ylabel('Accuracy')
    plt.title(f'Random Forest: {title_label}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/home/kjmetzler/random_forest_accuracy_{std_label}.png')
    plt.close()
    print_with_timestamp(f"Saved plot: random_forest_accuracy_{std_label}.png")
    
    # MLP
    print_with_timestamp(f"Running tests for MLP with {title_label}...")
    results_mlp = run_tests(real_data, syn_data, testing, num_points_per_class, ratios, train_and_evaluate_mlp)
    num_points_mlp, ratios_mlp, accuracies_mlp, num_datapoints_mlp = zip(*results_mlp)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ratios_mlp))))
    for i, ratio in enumerate(np.unique(ratios_mlp)):
        mask = np.array(ratios_mlp) == ratio
        plt.plot(np.array(num_points_mlp)[mask], np.array(accuracies_mlp)[mask], marker='o', label=f'Ratio {ratio:.1f}', color=colors[i])
    plt.xlabel('Total Number of Datapoints per Class')
    plt.ylabel('Accuracy')
    plt.title(f'MLP: {title_label}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/home/kjmetzler/mlp_accuracy_{std_label}.png')
    plt.close()
    print_with_timestamp(f"Saved plot: mlp_accuracy_{std_label}.png")

print_with_timestamp("All tests complete!")