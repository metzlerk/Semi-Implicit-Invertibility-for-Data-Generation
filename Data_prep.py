import pandas as pd

def create_subset(file_path, output_path, n=1000):
    # Read the feather file
    df = pd.read_feather(file_path)
    
    # Get the unique classes from the 'Label' column
    classes = df['Label'].unique()
    
    # Create an empty DataFrame to store the subset
    subset_df = pd.DataFrame()
    
    # For each class, sample n rows and append to the subset DataFrame
    for cls in classes:
        class_subset = df[df['Label'] == cls].sample(n=n, random_state=42)
        subset_df = pd.concat([subset_df, class_subset], ignore_index=True)
    
    # Save the subset to a new feather file
    subset_df.reset_index(drop=True).to_feather(output_path)
    
    # Clear the DataFrame from memory
    del df
    del subset_df

# Create subsets for each dataset
create_subset('/home/kjmetzler/test_data.feather', '/home/kjmetzler/test_data_subset.feather')
create_subset('/home/kjmetzler/train_data.feather', '/home/kjmetzler/train_data_subset.feather')
create_subset('/home/kjmetzler/synthetic_train_spectra.feather', '/home/kjmetzler/synthetic_train_spectra_subset.feather')