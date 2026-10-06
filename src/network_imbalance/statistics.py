import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def paired_comparison(results: pd.DataFrame, metric="minority_f1"):
    rows = []
    for model in results["model"].unique():
        sub = results[results.model == model]
        strategies = list(sub.strategy.unique())
        baseline = sub[sub.strategy == "baseline"].set_index("fold")[metric]
        for strategy in strategies:
            if strategy == "baseline":
                continue
            other = sub[sub.strategy == strategy].set_index("fold")[metric]
            joined = pd.concat([baseline, other], axis=1, join="inner").dropna()
            if len(joined) < 2:
                continue
            a, b = joined.iloc[:, 0], joined.iloc[:, 1]
            t_stat, t_p = ttest_rel(a, b)
            try:
                w_stat, w_p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            diff = b - a
            rows.append({
                "model": model,
                "comparison": f"{strategy} vs baseline",
                "metric": metric,
                "n_pairs": len(joined),
                "mean_difference": diff.mean(),
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            })
    return pd.DataFrame(rows)
