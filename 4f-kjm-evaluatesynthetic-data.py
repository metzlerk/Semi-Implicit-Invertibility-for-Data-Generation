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
cate_data = pd.read_feather('/home/kjmetzler/synthetic_train_spectra_subset.feather')
kevin_data = pd.read_csv('/home/kjmetzler/generated_synthetic_data.csv')
testing = pd.read_feather('/home/kjmetzler/test_data_subset.feather')
real_data = pd.read_feather('/home/kjmetzler/train_data_subset.feather')

# One-hot encoding
def one_hot_encode(df, label_col):
    one_hot_labels = pd.get_dummies(df[label_col]).astype(int)
    df.drop(label_col, axis=1, inplace=True)
    return df, one_hot_labels

print_with_timestamp("One-hot encoding test data...")
cate_data, cate_one_hot_labels = one_hot_encode(cate_data, 'Label')
cate_data = pd.concat([cate_data, cate_one_hot_labels], axis=1)

kevin_data, kevin_one_hot_labels = one_hot_encode(kevin_data, 'Label')
kevin_data = pd.concat([kevin_data, kevin_one_hot_labels], axis=1)
kevin_data = kevin_data.drop(columns=['1684','1683','1682','1681','1680','1679','1678','1677','1676'], axis=1)
# Rename the last 8 columns to the specified names
kevin_data.rename(columns={kevin_data.columns[-8]: 'DEB', 
                           kevin_data.columns[-7]: 'DEM', 
                           kevin_data.columns[-6]: 'DMMP', 
                           kevin_data.columns[-5]: 'DPM', 
                           kevin_data.columns[-4]: 'DtBP', 
                           kevin_data.columns[-3]: 'JP8', 
                           kevin_data.columns[-2]: 'MES', 
                           kevin_data.columns[-1]: 'TEPO'}, inplace=True)

testing = testing.drop(columns=['Unnamed: 0', 'index', 'Label'], axis=1)
real_data = real_data.drop(columns=['Unnamed: 0', 'index', 'Label'], axis=1)

label_size = cate_one_hot_labels.shape[1]
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

print_with_timestamp("Running tests for Random Forest with Cate's Synthetic Data...")
results_rf = run_tests(real_data, cate_data, testing, num_points_per_class, ratios, train_and_evaluate_rf)
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

print_with_timestamp("Running tests for MLP with Cate's Synthetic Data...")
results_mlp = run_tests(real_data, cate_data, testing, num_points_per_class, ratios, train_and_evaluate_mlp)
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

print_with_timestamp("Running tests for Random Forest with Kevin's Synthetic Data...")
results_rf_output = run_tests(real_data, kevin_data, testing, num_points_per_class, ratios, train_and_evaluate_rf)
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

print_with_timestamp("Running tests for MLP with Kevin's Synthetic Data...")
results_mlp_output = run_tests(real_data, kevin_data, testing, num_points_per_class, ratios, train_and_evaluate_mlp)
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