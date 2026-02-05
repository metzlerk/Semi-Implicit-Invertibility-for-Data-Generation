import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, 'train_data.feather')
train_df = pd.read_feather(train_path)

p_cols = [c for c in train_df.columns if c.startswith('p_')]
n_cols = [c for c in train_df.columns if c.startswith('n_')]
onehot_cols = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']

# IMS matrix and labels
X = np.concatenate([train_df[p_cols].values, train_df[n_cols].values], axis=1)
y_onehot = train_df[onehot_cols].values
labels = y_onehot.argmax(axis=1)
class_names = onehot_cols

# Standardize (mean-center) before PCA
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))
for i, cname in enumerate(class_names):
    mask = labels == i
    plt.scatter(X_pca[mask,0], X_pca[mask,1], s=8, alpha=0.6, color=colors[i], label=cname)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('IMS Train Set: PCA by Chemical Class')
plt.legend(markerscale=2, fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(IMAGES_DIR, 'ims_train_pca_by_class.png')
plt.savefig(out_path, dpi=150)
print(f'Saved {out_path}')
