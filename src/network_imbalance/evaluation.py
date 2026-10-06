from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, f1_score,
    precision_score, recall_score, average_precision_score, confusion_matrix,
)


def metric_row(y_true, y_pred, y_score, model, strategy, fold, classes):
    counts = np.bincount(y_true, minlength=len(classes))
    minority_idx = int(np.argmin(counts))
    pos_scores = y_score[:, minority_idx] if y_score is not None and y_score.ndim == 2 and y_score.shape[1] == len(classes) else None
    return {
        "model": model,
        "strategy": strategy,
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "minority_precision": precision_score(y_true, y_pred, labels=[minority_idx], average="macro", zero_division=0),
        "minority_recall": recall_score(y_true, y_pred, labels=[minority_idx], average="macro", zero_division=0),
        "minority_f1": f1_score(y_true, y_pred, labels=[minority_idx], average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y_true, pos_scores) if pos_scores is not None else np.nan,
    }


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary(results: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "macro_f1", "pr_auc",
               "minority_precision", "minority_recall", "minority_f1", "fit_seconds"]
    agg = {}
    for m in metrics:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
    agg["folds"] = ("fold", "count")
    summary = results.groupby(["model", "strategy"], as_index=False).agg(**agg)
    # Convenient aliases used by plotting code.
    for m in ["recall", "f1", "macro_f1", "balanced_accuracy", "pr_auc"]:
        summary[m] = summary[f"{m}_mean"]
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    return summary
