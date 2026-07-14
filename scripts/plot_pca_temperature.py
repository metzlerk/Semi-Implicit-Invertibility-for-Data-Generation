import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def find_dataframe():
    candidates = [
        '/scratch/kjmetzler/train_data_with_conditions.feather',
        'Data/train_data.feather',
        '/home/kjmetzler/train_data_with_conditions.feather',
        '/home/kjmetzler/train_data_subset.feather',
        '/home/kjmetzler/train_data.feather',
        'train_data.feather',
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_feather(p)
                print(f'Loaded dataframe from {p} with shape {df.shape}')
                return df
            except Exception as e:
                print(f'Found {p} but failed to read: {e}')
    raise FileNotFoundError('No train_data feather found in candidate paths')


def detect_temp_col(df):
    candidates = ['TemperatureKelvin', 'temp_K', 'Temperature_K', 'TempK', 'temperature']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any column name containing 'temp' or 'Temperature'
    for c in df.columns:
        if 'temp' in c.lower() or 'temperature' in c.lower():
            return c
    return None


def detect_label_column(df):
    # 1) explicit 'Label' column
    if 'Label' in df.columns:
        return 'Label', 'categorical'

    # 2) one-hot encoded labels detection: columns that are only 0/1
    bool_cols = []
    for c in df.columns:
        vals = df[c].dropna().unique()
        if set(vals).issubset({0, 1}):
            bool_cols.append(c)
    if len(bool_cols) >= 2:
        # check if these columns form a one-hot encoding for most rows
        sums = df[bool_cols].sum(axis=1)
        frac_one = (sums == 1).mean()
        if frac_one > 0.5:
            return bool_cols, 'onehot'

    # 3) categorical-like column: low cardinality and object dtype
    for c in df.columns:
        if df[c].dtype == object or df[c].dtype.name == 'category':
            if df[c].nunique() < 200:
                return c, 'categorical'

    # 4) fallback: try to find columns with prefix 'class_'
    class_cols = [c for c in df.columns if c.startswith('class_') or c.startswith('Class')]
    if class_cols:
        return class_cols, 'onehot'

    return None, None


def prepare_labels(df, label_info):
    label_cols, ltype = label_info
    if ltype == 'categorical':
        labels = df[label_cols].astype(str)
        return labels.values
    elif ltype == 'onehot':
        # label_cols is a list of columns
        lab = df[label_cols].values
        idx = np.argmax(lab, axis=1)
        names = [str(c) for c in label_cols]
        return np.array([names[i] for i in idx])
    else:
        raise ValueError('Unknown label type')


def run(plot_n_per_class=1000, max_classes=4, out_path='results/pca_temp_scatter.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = find_dataframe()

    temp_col = detect_temp_col(df)
    if temp_col is None:
        raise RuntimeError('No temperature column found in dataframe')

    label_info = detect_label_column(df)
    if label_info[0] is None:
        raise RuntimeError('No label column(s) detected')

    # figure out feature columns
    if label_info[1] == 'onehot':
        label_cols = label_info[0]
    else:
        label_cols = [label_info[0]]

    feature_cols = [c for c in df.columns if c not in label_cols + [temp_col]]
    # keep only numeric features
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    print(f'Using {len(feature_cols)} feature columns for PCA')

    labels = None
    if label_info[1] == 'onehot':
        labels = prepare_labels(df, label_info)
    else:
        labels = df[label_cols[0]].astype(str).values

    # select top classes by count
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    selected_classes = unique[order][:max_classes]
    print('Selected classes:', selected_classes)

    samples = []
    sample_labels = []
    sample_temps = []
    for cls in selected_classes:
        idxs = np.where(labels == cls)[0]
        if len(idxs) == 0:
            continue
        take = min(plot_n_per_class, len(idxs))
        chosen = np.random.choice(idxs, size=take, replace=False)
        samples.append(df.iloc[chosen][feature_cols].values)
        sample_labels.extend([cls] * take)
        sample_temps.extend(df.iloc[chosen][temp_col].values)

    X = np.vstack(samples)
    sample_labels = np.array(sample_labels)
    sample_temps = np.array(sample_temps)

    # scale and PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)

    # 3D scatter
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_classes)))
    for i, cls in enumerate(selected_classes):
        mask = sample_labels == cls
        ax.scatter(pcs[mask, 0], pcs[mask, 1], sample_temps[mask], label=str(cls), color=colors[i], s=6, alpha=0.7)

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel(temp_col)
    ax.set_xlim(left=float(np.min(pcs[:, 0])), right=50)
    ax.set_ylim(bottom=-100, top=float(np.max(pcs[:, 1])))
    ax.set_title('PCA (dim1, dim2) vs Temperature')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved PCA vs Temperature scatter to {out_path}')


if __name__ == '__main__':
    run()
