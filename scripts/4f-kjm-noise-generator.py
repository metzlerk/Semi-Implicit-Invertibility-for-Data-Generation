import torch
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train = pd.read_feather('/home/kjmetzler/train_data_subset.feather')
print('Shape of training data:',train.shape)

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
    
    combined_df = pd.concat([pd.DataFrame(blurred_chemicals_combined), pd.DataFrame(blurred_labels_combined), pd.DataFrame(blur_counts_column, columns=['Blur_Count'])], axis=1, ignore_index=True)
    combined_df.columns = [f'col_{i}' for i in range(combined_df.shape[1])]
    return combined_df

# Pad the dataframe
df_padded = df

# Add noise to the padded dataframe
n = 10
noisy_df = add_noise(df_padded, n)

# Save the noisy dataframe as a feather file
noisy_df.to_feather('/home/kjmetzler/noisy_data.feather')