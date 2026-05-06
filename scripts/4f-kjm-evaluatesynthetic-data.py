import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_NAMES = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
start_time = time.time()


def elapsed_time():
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))


def print_with_timestamp(message):
    print(f"{elapsed_time()}: {message}", flush=True)


def load_synthetic_data(spectra_path, labels_path):
    spectra = np.load(spectra_path)
    labels = np.load(labels_path)

    df = pd.DataFrame(spectra)
    one_hot = np.zeros((len(labels), len(CLASS_NAMES)))
    for i, label in enumerate(labels):
        one_hot[i, label] = 1
    one_hot_df = pd.DataFrame(one_hot, columns=CLASS_NAMES)

    return pd.concat([df, one_hot_df], axis=1)


def one_hot_from_label_series(label_series):
    if pd.api.types.is_numeric_dtype(label_series):
        label_indices = label_series.to_numpy(dtype=int)
    else:
        label_values = label_series.astype(str).to_numpy()
        if np.all(np.isin(label_values, CLASS_NAMES)):
            one_hot_df = pd.get_dummies(
                pd.Categorical(label_values, categories=CLASS_NAMES)
            ).astype(int)
            one_hot_df.columns = CLASS_NAMES
            return one_hot_df
        try:
            label_indices = label_series.astype(int).to_numpy()
        except ValueError as exc:
            raise ValueError(
                "Unsupported label encoding. Expected integer class IDs or "
                f"class names in {CLASS_NAMES}."
            ) from exc

    if np.any((label_indices < 0) | (label_indices >= len(CLASS_NAMES))):
        raise ValueError(
            f"Label indices out of range. Expected values in [0, {len(CLASS_NAMES) - 1}]."
        )

    one_hot = np.zeros((len(label_indices), len(CLASS_NAMES)), dtype=int)
    one_hot[np.arange(len(label_indices)), label_indices] = 1
    return pd.DataFrame(one_hot, columns=CLASS_NAMES)


def load_synthetic_data_with_embedded_labels(data_path, label_column=None):
    suffix = Path(data_path).suffix.lower()
    if suffix == '.feather':
        df = pd.read_feather(data_path)
    elif suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file type for synthetic dataset: {data_path}")

    for col in ['Unnamed: 0', 'index']:
        if col in df.columns:
            df = df.drop(columns=col)

    label_columns_present = [col for col in CLASS_NAMES if col in df.columns]
    if label_columns_present:
        labels_df = df[label_columns_present].copy().astype(int)
        for class_name in CLASS_NAMES:
            if class_name not in labels_df.columns:
                labels_df[class_name] = 0
        labels_df = labels_df[CLASS_NAMES]
        features_df = df.drop(columns=label_columns_present)
        return pd.concat(
            [features_df.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1
        )

    candidate_label_columns = []
    if label_column is not None:
        candidate_label_columns.append(label_column)
    candidate_label_columns.extend(['Label', 'Label_U'])

    selected_label_col = next((c for c in candidate_label_columns if c in df.columns), None)
    if selected_label_col is None:
        raise ValueError(
            f"Could not infer labels for {data_path}. Expected one of CLASS_NAMES as columns "
            "or a label column named 'Label'/'Label_U'."
        )

    labels_df = one_hot_from_label_series(df[selected_label_col])
    features_df = df.drop(columns=[selected_label_col])
    return pd.concat(
        [features_df.reset_index(drop=True), labels_df.reset_index(drop=True)], axis=1
    )


def build_rf(params=None, random_state=42):
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    if params:
        rf.set_params(**params)
    return rf


def build_mlp(params=None, random_state=42):
    mlp = MLPClassifier(
        hidden_layer_sizes=(1000, 500, 250, 100),
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        max_iter=10000,
        random_state=random_state,
    )
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', mlp)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def build_svm(params=None, random_state=42):
    svm = SVC(kernel='rbf')
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', svm)])
    if params:
        pipeline.set_params(**params)
    return pipeline


def train_and_evaluate(build_estimator, params, X_train, y_train, X_test, y_test, random_state):
    model = build_estimator(params=params, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def build_training_set(X_real, y_real, X_synthetic, y_synthetic, num_points, ratio):
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
    if not X_train or not y_train:
        return None, None
    return np.vstack(X_train), np.hstack(y_train)


def run_tests(
    real_df,
    synthetic_df,
    test_df,
    num_points_per_class,
    ratios,
    build_estimator,
    estimator_params,
    random_state,
    label_size,
    data_size,
):
    X_real = real_df.iloc[:, :data_size].values
    y_real = np.argmax(real_df.iloc[:, -label_size:].values, axis=1)
    X_synthetic = synthetic_df.iloc[:, :data_size].values
    y_synthetic = np.argmax(synthetic_df.iloc[:, -label_size:].values, axis=1)
    X_test = test_df.iloc[:, :data_size].values
    y_test = np.argmax(test_df.iloc[:, -label_size:].values, axis=1)
    results = []

    for ratio in ratios:
        for num_points in num_points_per_class:
            X_train, y_train = build_training_set(
                X_real, y_real, X_synthetic, y_synthetic, num_points, ratio
            )
            if X_train is not None and y_train is not None:
                accuracy = train_and_evaluate(
                    build_estimator,
                    estimator_params,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    random_state,
                )
                results.append((num_points, ratio, accuracy, len(X_train)))
    return results


def build_results_df(results, classifier_name, std_label):
    num_points, ratios_col, accuracies, train_counts = zip(*results)
    return pd.DataFrame(
        {
            'classifier': classifier_name,
            'std_label': std_label,
            'num_points_per_class': num_points,
            'real_ratio': ratios_col,
            'accuracy': accuracies,
            'train_datapoints': train_counts,
        }
    )


def summarize_against_real(results_df, classifier_name, std_label):
    deltas = []
    synthetic_only_better = 0
    improved_points = 0

    for num_points in sorted(results_df['num_points_per_class'].unique()):
        point_df = results_df[results_df['num_points_per_class'] == num_points]
        real_only = point_df.loc[np.isclose(point_df['real_ratio'], 1.0), 'accuracy']
        synthetic_only = point_df.loc[np.isclose(point_df['real_ratio'], 0.0), 'accuracy']
        if real_only.empty:
            continue

        real_acc = float(real_only.iloc[0])
        best_idx = point_df['accuracy'].idxmax()
        best_row = point_df.loc[best_idx]
        delta = float(best_row['accuracy']) - real_acc
        deltas.append(delta)

        if float(best_row['accuracy']) > real_acc:
            improved_points += 1
        if not synthetic_only.empty and float(synthetic_only.iloc[0]) > real_acc:
            synthetic_only_better += 1

    if not deltas:
        print_with_timestamp(
            f"SUMMARY classifier={classifier_name} std={std_label}: no valid points found"
        )
        return

    print_with_timestamp(
        "SUMMARY "
        f"classifier={classifier_name} "
        f"std={std_label} "
        f"mean_best_minus_real={np.mean(deltas):+.4f} "
        f"median_best_minus_real={np.median(deltas):+.4f} "
        f"improved_points={improved_points}/{len(deltas)} "
        f"synthetic_only_better={synthetic_only_better}/{len(deltas)}"
    )


def slugify(label):
    cleaned = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(label))
    return '_'.join(filter(None, cleaned.split('_')))


def parse_classifier_list(raw_value):
    requested = [item.strip() for item in raw_value.split(',') if item.strip()]
    return requested


def get_classifier_specs():
    return {
        'RandomForest': {
            'builder': build_rf,
            'tune_space': {
                'n_estimators': [200, 400, 800],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
            },
        },
        'MLP': {
            'builder': build_mlp,
            'tune_space': {
                'model__hidden_layer_sizes': [
                    (512,),
                    (512, 256),
                    (1000, 500, 250, 100),
                    (512, 256, 128),
                    (256, 128),
                ],
                'model__alpha': [1e-4, 1e-3, 1e-2],
                'model__learning_rate_init': [1e-4, 3e-4, 1e-3],
                'model__activation': ['relu', 'tanh'],
            },
        },
        'SVM': {
            'builder': build_svm,
            'tune_space': {
                'model__C': [0.1, 1.0, 10.0, 100.0],
                'model__gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
            },
        },
    }


def tune_classifier(
    classifier_name,
    build_estimator,
    param_distributions,
    X_train,
    y_train,
    args,
):
    if not param_distributions:
        return None

    class_counts = np.bincount(y_train)
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    cv_splits = min(args.tune_cv, min_class_count)
    if cv_splits < 2:
        print_with_timestamp(
            f"Tuning skipped for {classifier_name}: not enough samples per class "
            f"(min={min_class_count})."
        )
        return None

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    search = RandomizedSearchCV(
        estimator=build_estimator(params=None, random_state=args.random_state),
        param_distributions=param_distributions,
        n_iter=args.tune_iterations,
        scoring='accuracy',
        n_jobs=args.tune_jobs,
        cv=cv,
        random_state=args.random_state,
        refit=True,
    )
    print_with_timestamp(
        f"Tuning {classifier_name} with {args.tune_iterations} iterations "
        f"(cv={cv_splits})..."
    )
    search.fit(X_train, y_train)
    print_with_timestamp(f"Best params for {classifier_name}: {search.best_params_}")
    return search.best_params_


def save_plot(results, title, out_path):
    import matplotlib.pyplot as plt

    num_points_arr, ratios_arr, accuracies_arr, _ = zip(*results)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(ratios_arr))))
    for i, ratio in enumerate(np.unique(ratios_arr)):
        mask = np.array(ratios_arr) == ratio
        plt.plot(
            np.array(num_points_arr)[mask],
            np.array(accuracies_arr)[mask],
            marker='o',
            label=f'Ratio {ratio:.1f}',
            color=colors[i],
        )
    plt.xlabel('Total Number of Datapoints per Class')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate synthetic data impact on classifiers with optional tuning.'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving PNG plots and only compute numeric results.',
    )
    parser.add_argument(
        '--csv-path',
        default='/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/eval_synthetic_metrics.csv',
        help='Path to save detailed evaluation metrics CSV.',
    )
    parser.add_argument(
        '--classifiers',
        default='RandomForest,MLP',
        help='Comma-separated list of classifiers to evaluate (RandomForest, MLP, SVM).',
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Run hyperparameter search for each classifier and synthetic dataset.',
    )
    parser.add_argument(
        '--tune-num-points',
        type=int,
        default=20,
        help='Points per class used to build the tuning dataset.',
    )
    parser.add_argument(
        '--tune-ratio',
        type=float,
        default=0.5,
        help='Real-data ratio used to build the tuning dataset.',
    )
    parser.add_argument(
        '--tune-iterations',
        type=int,
        default=24,
        help='Randomized search iterations per classifier.',
    )
    parser.add_argument(
        '--tune-cv',
        type=int,
        default=3,
        help='Maximum CV folds for hyperparameter search.',
    )
    parser.add_argument(
        '--tune-jobs',
        type=int,
        default=-1,
        help='Parallel jobs for hyperparameter search.',
    )
    parser.add_argument(
        '--tuned-suffix',
        default='_tuned',
        help='Suffix appended to classifier name when tuning is enabled.',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for classifiers and tuning.',
    )
    parser.add_argument(
        '--synthetic-std1-spectra',
        default='/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.0.npy',
        help='Path to synthetic spectra for std=1.0.',
    )
    parser.add_argument(
        '--synthetic-std1-labels',
        default='/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.0.npy',
        help='Path to synthetic labels for std=1.0.',
    )
    parser.add_argument(
        '--synthetic-std1p5-spectra',
        default='/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.5.npy',
        help='Path to synthetic spectra for std=1.5.',
    )
    parser.add_argument(
        '--synthetic-std1p5-labels',
        default='/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.5.npy',
        help='Path to synthetic labels for std=1.5.',
    )
    parser.add_argument(
        '--synthetic-std2-spectra',
        default='/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std2.0.npy',
        help='Path to synthetic spectra for std=2.0.',
    )
    parser.add_argument(
        '--synthetic-std2-labels',
        default='/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std2.0.npy',
        help='Path to synthetic labels for std=2.0.',
    )
    parser.add_argument(
        '--cate-path',
        default='/home/kjmetzler/scratch/CARL/universal_generator/_synthetic_test_spectra.feather',
        help='Path to CATE synthetic spectra (feather or csv).',
    )
    parser.add_argument(
        '--skip-cate',
        action='store_true',
        help='Skip loading the CATE synthetic dataset.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print_with_timestamp("Loading data...")
    testing = pd.read_feather('/home/kjmetzler/test_data_subset.feather')
    real_data = pd.read_feather('/home/kjmetzler/train_data_subset.feather')

    print_with_timestamp("Loading synthetic datasets...")
    std1_data = load_synthetic_data(
        args.synthetic_std1_spectra,
        args.synthetic_std1_labels,
    )
    std15_data = load_synthetic_data(
        args.synthetic_std1p5_spectra,
        args.synthetic_std1p5_labels,
    )
    std2_data = load_synthetic_data(
        args.synthetic_std2_spectra,
        args.synthetic_std2_labels,
    )
    cate_data = None
    if not args.skip_cate:
        cate_data = load_synthetic_data_with_embedded_labels(args.cate_path)

    testing = testing.drop(columns=['Unnamed: 0', 'index', 'Label'], errors='ignore')
    real_data = real_data.drop(columns=['Unnamed: 0', 'index', 'Label'], errors='ignore')

    label_size = len(CLASS_NAMES)
    data_size = real_data.shape[1] - label_size

    num_points_per_class = np.arange(1, 51)
    ratios = np.linspace(0, 1, 11)

    synthetic_datasets = [
        (std1_data, 'std1.0', 'Synthetic Data (std=1.0)'),
        (std15_data, 'std1.5', 'Synthetic Data (std=1.5)'),
        (std2_data, 'std2.0', 'Synthetic Data (std=2.0)'),
    ]
    if cate_data is not None:
        synthetic_datasets.append((cate_data, 'cate', 'Synthetic Data (CATE)'))

    if not (0.0 <= args.tune_ratio <= 1.0):
        raise ValueError("--tune-ratio must be between 0.0 and 1.0.")
    if args.tune_num_points < 1:
        raise ValueError("--tune-num-points must be at least 1.")

    classifier_specs = get_classifier_specs()
    requested_classifiers = parse_classifier_list(args.classifiers)
    unknown = [name for name in requested_classifiers if name not in classifier_specs]
    if unknown:
        raise ValueError(
            f"Unknown classifiers requested: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(classifier_specs))}"
        )

    all_results = []

    for syn_data, std_label, title_label in synthetic_datasets:
        tuned_params = {}
        if args.tune:
            print_with_timestamp(
                f"Preparing tuning set (n={args.tune_num_points}, ratio={args.tune_ratio:.2f}) "
                f"for {title_label}..."
            )
            X_real = real_data.iloc[:, :data_size].values
            y_real = np.argmax(real_data.iloc[:, -label_size:].values, axis=1)
            X_synthetic = syn_data.iloc[:, :data_size].values
            y_synthetic = np.argmax(syn_data.iloc[:, -label_size:].values, axis=1)
            X_tune, y_tune = build_training_set(
                X_real, y_real, X_synthetic, y_synthetic, args.tune_num_points, args.tune_ratio
            )
            if X_tune is None or y_tune is None:
                print_with_timestamp("Skipping tuning: could not build tuning dataset.")
            else:
                for classifier_name in requested_classifiers:
                    spec = classifier_specs[classifier_name]
                    tuned_params[classifier_name] = tune_classifier(
                        classifier_name,
                        spec['builder'],
                        spec['tune_space'],
                        X_tune,
                        y_tune,
                        args,
                    )

        for classifier_name in requested_classifiers:
            spec = classifier_specs[classifier_name]
            params = tuned_params.get(classifier_name)
            classifier_label = (
                f"{classifier_name}{args.tuned_suffix}" if args.tune else classifier_name
            )
            print_with_timestamp(
                f"Running tests for {classifier_label} with {title_label}..."
            )
            results = run_tests(
                real_data,
                syn_data,
                testing,
                num_points_per_class,
                ratios,
                spec['builder'],
                params,
                args.random_state,
                label_size,
                data_size,
            )
            results_df = build_results_df(results, classifier_label, std_label)
            all_results.append(results_df)
            summarize_against_real(results_df, classifier_label, std_label)

            if not args.no_plots:
                plot_name = f"{slugify(classifier_label)}_accuracy_{std_label}.png"
                save_plot(
                    results,
                    f'{classifier_label}: {title_label}',
                    f'/home/kjmetzler/{plot_name}',
                )
                print_with_timestamp(f"Saved plot: {plot_name}")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(args.csv_path, index=False)
    print_with_timestamp(f"Saved metrics CSV: {args.csv_path}")
    print_with_timestamp("All tests complete!")


if __name__ == '__main__':
    main()
