from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from imblearn.pipeline import Pipeline

from .config import load_config
from .data_loader import load_csv, make_demo_dataset
from .preprocessing import preprocessing_steps, clean_target
from .feature_selection import make_selector
from .imbalance import make_sampler
from .models import make_model
from .evaluation import metric_row, save_summary, save_confusion_matrix
from .statistics import paired_comparison
from .visualization import plot_metric, plot_class_distribution


def build_pipeline(strategy, model_name, seed, n_jobs, n_features, fs_enabled, fs_k):
    standardize = strategy != "baseline"
    feature_selected = strategy in {"feature_selected", "combined"}
    weighted = strategy in {"class_weighted", "combined"}
    sampler = make_sampler(strategy, seed)
    steps = preprocessing_steps(standardize=standardize)
    if feature_selected and fs_enabled:
        k = min(fs_k, n_features)
        steps.append(("selector", make_selector(k, seed)))
    if sampler != "passthrough":
        steps.append(("sampler", sampler))
    steps.append(("model", make_model(model_name, weighted, seed, n_jobs)))
    return Pipeline(steps)


def run(config_path="configs/experiment.yaml"):
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg.seed)
    if cfg.data_path.exists():
        X, y, meta = load_csv(cfg.data_path, cfg.target_column, cfg.drop_columns, cfg.max_rows)
    elif cfg.use_demo_if_missing:
        X, y, meta = make_demo_dataset(cfg.demo_samples, cfg.seed)
    else:
        raise FileNotFoundError(f"Dataset not found: {cfg.data_path}")
    y = clean_target(y)

    root = Path("results")
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    plot_class_distribution(y, root / "figures" / "class_distribution.png", "Input class distribution")
    with open(root / "tables" / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump({**meta, "class_counts": y.value_counts().to_dict()}, f, indent=2)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    records = []
    timings = []
    predictions = {}

    for model_name in cfg.models:
        for strategy in cfg.strategies:
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_encoded), start=1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                pipe = build_pipeline(strategy, model_name, cfg.seed, cfg.n_jobs, X.shape[1], cfg.feature_selection_enabled, cfg.feature_selection_k)
                start = time.perf_counter()
                pipe.fit(X_train, y_train)
                elapsed = time.perf_counter() - start
                pred = pipe.predict(X_test)
                proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
                row = metric_row(y_test, pred, proba, model_name, strategy, fold, le.classes_)
                row["fit_seconds"] = elapsed
                row["train_rows"] = len(train_idx)
                row["test_rows"] = len(test_idx)
                records.append(row)
                timings.append(row)
                predictions.setdefault((model_name, strategy), {"y_true": [], "y_pred": []})
                predictions[(model_name, strategy)]["y_true"].extend(y_test.tolist())
                predictions[(model_name, strategy)]["y_pred"].extend(pred.tolist())

    results = pd.DataFrame(records)
    results.to_csv(root / "tables" / "fold_results.csv", index=False)
    summary = save_summary(results, root / "tables")
    stats = paired_comparison(results, metric="minority_f1")
    stats.to_csv(root / "tables" / "paired_statistics_minority_f1.csv", index=False)
    for (model_name, strategy), pred_data in predictions.items():
        save_confusion_matrix(
            pred_data["y_true"], pred_data["y_pred"],
            list(range(len(le.classes_))),
            f"{model_name} — {strategy}",
            root / "figures" / f"confusion_{model_name}_{strategy}.png",
        )
    for metric in ["recall", "f1", "macro_f1", "balanced_accuracy", "pr_auc"]:
        plot_metric(summary, metric, root / "figures" / f"{metric}.png")
    print("Experiment complete.")
    print(summary.sort_values(["model", "minority_f1_mean"], ascending=[True, False]).to_string(index=False))
    return results, summary, stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the IDS preprocessing and class-imbalance study.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
