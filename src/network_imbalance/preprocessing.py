from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocessing_steps(standardize=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    return steps


def clean_target(y):
    y = y.astype(str).str.strip()
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    return y
