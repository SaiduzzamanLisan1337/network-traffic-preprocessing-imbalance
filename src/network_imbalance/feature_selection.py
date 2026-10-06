from sklearn.feature_selection import SelectKBest, mutual_info_classif


def make_selector(k: int, seed: int):
    return SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, random_state=seed), k=k)
