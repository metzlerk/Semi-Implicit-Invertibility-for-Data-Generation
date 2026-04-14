#!/usr/bin/env python3
"""
Extract raw data for fig_gain_over_real_only.png and fig_optimal_real_ratio.png
from the evaluation metrics CSV file.
"""

import pandas as pd
import numpy as np

# Load the evaluation metrics
csv_path = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/eval_synthetic_metrics_summary.csv'
df = pd.read_csv(csv_path)

print("Loaded evaluation metrics with shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nUnique classifiers:", df['classifier'].unique())
print("Unique std_labels:", df['std_label'].unique())
print("Points per class range:", df['num_points_per_class'].min(), "to", df['num_points_per_class'].max())
print("Real ratio range:", df['real_ratio'].min(), "to", df['real_ratio'].max())

# ==============================================================================
# FIGURE 1: Gain over real-only training
# For each (classifier, std_label, num_points_per_class), compute:
# best_gain = max_r(Acc(r)) - Acc(r=1.0)
# ==============================================================================

gain_data = []

for classifier in df['classifier'].unique():
    for std_label in df['std_label'].unique():
        for num_points in sorted(df['num_points_per_class'].unique()):
            # Filter to this specific configuration
            subset = df[
                (df['classifier'] == classifier) &
                (df['std_label'] == std_label) &
                (df['num_points_per_class'] == num_points)
            ]

            if subset.empty:
                continue

            # Get real-only accuracy (r=1.0)
            real_only = subset[np.isclose(subset['real_ratio'], 1.0)]
            if real_only.empty:
                continue

            real_only_acc = float(real_only['accuracy'].iloc[0])

            # Get best accuracy across all ratios
            best_acc = float(subset['accuracy'].max())

            # Compute gain
            gain = best_acc - real_only_acc

            gain_data.append({
                'classifier': classifier,
                'std_label': std_label,
                'num_points_per_class': num_points,
                'real_only_accuracy': real_only_acc,
                'best_accuracy': best_acc,
                'gain_over_real_only': gain
            })

gain_df = pd.DataFrame(gain_data)
gain_output = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/plot_data_gain_over_real_only.csv'
gain_df.to_csv(gain_output, index=False)
print(f"\n✓ Saved gain data to: {gain_output}")
print(f"  Shape: {gain_df.shape}")
print(f"\nSample rows:")
print(gain_df.head(10))

# ==============================================================================
# FIGURE 2: Optimal real ratio
# For each (classifier, std_label, num_points_per_class), find:
# optimal_ratio = argmax_r(Acc(r))
# ==============================================================================

optimal_ratio_data = []

for classifier in df['classifier'].unique():
    for std_label in df['std_label'].unique():
        for num_points in sorted(df['num_points_per_class'].unique()):
            # Filter to this specific configuration
            subset = df[
                (df['classifier'] == classifier) &
                (df['std_label'] == std_label) &
                (df['num_points_per_class'] == num_points)
            ]

            if subset.empty:
                continue

            # Find the ratio that gives maximum accuracy
            best_idx = subset['accuracy'].idxmax()
            best_row = subset.loc[best_idx]

            optimal_ratio_data.append({
                'classifier': classifier,
                'std_label': std_label,
                'num_points_per_class': num_points,
                'optimal_real_ratio': float(best_row['real_ratio']),
                'accuracy_at_optimal': float(best_row['accuracy'])
            })

optimal_df = pd.DataFrame(optimal_ratio_data)
optimal_output = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/plot_data_optimal_real_ratio.csv'
optimal_df.to_csv(optimal_output, index=False)
print(f"\n✓ Saved optimal ratio data to: {optimal_output}")
print(f"  Shape: {optimal_df.shape}")
print(f"\nSample rows:")
print(optimal_df.head(10))

# ==============================================================================
# Also save the full raw evaluation metrics for reference
# ==============================================================================

full_output = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/plot_data_full_evaluation_metrics.csv'
df.to_csv(full_output, index=False)
print(f"\n✓ Saved full evaluation data to: {full_output}")
print(f"  Shape: {df.shape}")

print("\n" + "="*80)
print("SUMMARY: Generated 3 CSV files:")
print("="*80)
print(f"1. {gain_output}")
print(f"   → Data for fig_gain_over_real_only.png")
print(f"   → Columns: classifier, std_label, num_points_per_class, real_only_accuracy, best_accuracy, gain_over_real_only")
print(f"\n2. {optimal_output}")
print(f"   → Data for fig_optimal_real_ratio.png")
print(f"   → Columns: classifier, std_label, num_points_per_class, optimal_real_ratio, accuracy_at_optimal")
print(f"\n3. {full_output}")
print(f"   → Complete evaluation metrics (all configurations)")
print(f"   → Columns: classifier, std_label, num_points_per_class, real_ratio, accuracy, train_datapoints")
print("="*80)
