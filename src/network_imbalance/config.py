from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    seed: int
    data_path: Path
    target_column: str
    drop_columns: list[str]
    max_rows: int | None
    test_size: float
    cv_folds: int
    n_jobs: int
    save_models: bool
    use_demo_if_missing: bool
    demo_samples: int
    feature_selection_enabled: bool
    feature_selection_method: str
    feature_selection_k: int
    models: list[str]
    strategies: list[str]


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    return Config(
        seed=int(c["seed"]),
        data_path=Path(c["data"]["path"]),
        target_column=c["data"]["target_column"],
        drop_columns=list(c["data"].get("drop_columns", [])),
        max_rows=c["data"].get("max_rows"),
        test_size=float(c["split"]["test_size"]),
        cv_folds=int(c["experiment"]["cv_folds"]),
        n_jobs=int(c["experiment"]["n_jobs"]),
        save_models=bool(c["experiment"]["save_models"]),
        use_demo_if_missing=bool(c["experiment"].get("use_demo_if_missing", True)),
        demo_samples=int(c["experiment"].get("demo_samples", 12000)),
        feature_selection_enabled=bool(c["feature_selection"]["enabled"]),
        feature_selection_method=c["feature_selection"]["method"],
        feature_selection_k=int(c["feature_selection"]["k"]),
        models=list(c["models"]),
        strategies=list(c["strategies"]),
    )
