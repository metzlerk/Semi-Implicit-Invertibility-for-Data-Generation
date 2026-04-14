#!/usr/bin/env python3
"""
Extract raw data for fig_gain_over_real_only.png and fig_optimal_real_ratio.png
using only standard library (no pandas/numpy dependencies).
"""

import csv
from collections import defaultdict

# Load the evaluation metrics
csv_path = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/eval_synthetic_metrics_summary.csv'

print("Reading evaluation metrics from:", csv_path)

# Read CSV and organize data
data = []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'classifier': row['classifier'],
            'std_label': row['std_label'],
            'num_points_per_class': int(row['num_points_per_class']),
            'real_ratio': float(row['real_ratio']),
            'accuracy': float(row['accuracy']),
            'train_datapoints': int(row['train_datapoints'])
        })

print(f"Loaded {len(data)} rows")

# Get unique values
classifiers = sorted(set(row['classifier'] for row in data))
std_labels = sorted(set(row['std_label'] for row in data))
num_points_list = sorted(set(row['num_points_per_class'] for row in data))

print(f"Classifiers: {classifiers}")
print(f"Std labels: {std_labels}")
print(f"Points per class: {min(num_points_list)} to {max(num_points_list)}")

# ==============================================================================
# FIGURE 1: Gain over real-only training
# ==============================================================================

gain_data = []

for classifier in classifiers:
    for std_label in std_labels:
        for num_points in num_points_list:
            # Filter to this configuration
            subset = [r for r in data if
                      r['classifier'] == classifier and
                      r['std_label'] == std_label and
                      r['num_points_per_class'] == num_points]

            if not subset:
                continue

            # Get real-only accuracy (r=1.0)
            real_only = [r for r in subset if abs(r['real_ratio'] - 1.0) < 0.01]
            if not real_only:
                continue

            real_only_acc = real_only[0]['accuracy']

            # Get best accuracy
            best_acc = max(r['accuracy'] for r in subset)

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

# Save gain data
gain_output = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/plot_data_gain_over_real_only.csv'
with open(gain_output, 'w', newline='') as f:
    fieldnames = ['classifier', 'std_label', 'num_points_per_class', 'real_only_accuracy', 'best_accuracy', 'gain_over_real_only']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(gain_data)

print(f"\n✓ Saved gain data to: {gain_output}")
print(f"  Rows: {len(gain_data)}")
print(f"  Columns: classifier, std_label, num_points_per_class, real_only_accuracy, best_accuracy, gain_over_real_only")

# ==============================================================================
# FIGURE 2: Optimal real ratio
# ==============================================================================

optimal_ratio_data = []

for classifier in classifiers:
    for std_label in std_labels:
        for num_points in num_points_list:
            # Filter to this configuration
            subset = [r for r in data if
                      r['classifier'] == classifier and
                      r['std_label'] == std_label and
                      r['num_points_per_class'] == num_points]

            if not subset:
                continue

            # Find the ratio that gives maximum accuracy
            best_row = max(subset, key=lambda x: x['accuracy'])

            optimal_ratio_data.append({
                'classifier': classifier,
                'std_label': std_label,
                'num_points_per_class': num_points,
                'optimal_real_ratio': best_row['real_ratio'],
                'accuracy_at_optimal': best_row['accuracy']
            })

# Save optimal ratio data
optimal_output = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/plot_data_optimal_real_ratio.csv'
with open(optimal_output, 'w', newline='') as f:
    fieldnames = ['classifier', 'std_label', 'num_points_per_class', 'optimal_real_ratio', 'accuracy_at_optimal']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(optimal_ratio_data)

print(f"\n✓ Saved optimal ratio data to: {optimal_output}")
print(f"  Rows: {len(optimal_ratio_data)}")
print(f"  Columns: classifier, std_label, num_points_per_class, optimal_real_ratio, accuracy_at_optimal")

# ==============================================================================
# Also copy the full raw evaluation metrics for reference
# ==============================================================================

import shutil
full_output = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/plot_data_full_evaluation_metrics.csv'
shutil.copy(csv_path, full_output)
print(f"\n✓ Copied full evaluation data to: {full_output}")
print(f"  Rows: {len(data)}")
print(f"  Columns: classifier, std_label, num_points_per_class, real_ratio, accuracy, train_datapoints")

print("\n" + "="*80)
print("SUMMARY: Generated 3 CSV files with raw plot data:")
print("="*80)
print(f"\n1. {gain_output}")
print(f"   → Data for fig_gain_over_real_only.png")
print(f"   → {len(gain_data)} rows")
print(f"   → Shows: For each budget, gain = best_accuracy - real_only_accuracy")
print(f"\n2. {optimal_output}")
print(f"   → Data for fig_optimal_real_ratio.png")
print(f"   → {len(optimal_ratio_data)} rows")
print(f"   → Shows: For each budget, the real_ratio that maximizes accuracy")
print(f"\n3. {full_output}")
print(f"   → Complete evaluation metrics (all configurations)")
print(f"   → {len(data)} rows")
print(f"   → All combinations of: classifier × std_label × num_points × real_ratio")
print("="*80)

# Show sample from gain data
print("\nSample from gain data (first 5 rows):")
for i, row in enumerate(gain_data[:5]):
    print(f"  {row}")

print("\nSample from optimal ratio data (first 5 rows):")
for i, row in enumerate(optimal_ratio_data[:5]):
    print(f"  {row}")
