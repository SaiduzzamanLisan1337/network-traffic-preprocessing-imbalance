from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(summary: pd.DataFrame, metric: str, out: Path):
    pivot = summary.pivot(index="strategy", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Strategy")
    ax.set_title(f"{metric.replace('_', ' ').title()} by preprocessing / imbalance strategy")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_class_distribution(y, out: Path, title="Class distribution"):
    counts = pd.Series(y).value_counts()
    ax = counts.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
