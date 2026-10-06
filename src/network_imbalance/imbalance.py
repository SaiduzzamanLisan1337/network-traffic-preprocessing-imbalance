from imblearn.over_sampling import SMOTE


def make_sampler(strategy: str, seed: int):
    return SMOTE(random_state=seed) if strategy in {"smote", "combined"} else "passthrough"
