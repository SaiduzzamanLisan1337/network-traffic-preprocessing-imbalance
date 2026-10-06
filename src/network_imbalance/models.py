from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def make_model(name: str, weighted: bool, seed: int, n_jobs: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1500, class_weight="balanced" if weighted else None, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced" if weighted else None,
            random_state=seed,
            n_jobs=n_jobs,
            min_samples_leaf=2,
        )
    raise ValueError(f"Unknown model: {name}")
