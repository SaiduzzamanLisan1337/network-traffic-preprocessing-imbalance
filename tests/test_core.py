import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from network_imbalance.data_loader import make_demo_dataset
from network_imbalance.models import make_model
from network_imbalance.imbalance import make_sampler
from network_imbalance.evaluation import metric_row


def test_demo_dataset_is_imbalanced():
    X, y, meta = make_demo_dataset(1000, 42)
    assert len(X) == len(y) == 1000
    assert y.value_counts().min() < y.value_counts().max()


def test_smote_factory():
    sampler = make_sampler("smote", 42)
    assert sampler.random_state == 42


def test_model_factory():
    model = make_model("random_forest", True, 42, 1)
    assert model.class_weight == "balanced"


def test_minority_metrics_are_reported():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])
    proba = np.array([[.9,.1],[.8,.2],[.7,.3],[.4,.6],[.2,.8],[.7,.3]])
    row = metric_row(y_true, y_pred, proba, "test", "baseline", 1, np.array(["ATTACK", "BENIGN"]))
    assert 0 <= row["minority_recall"] <= 1
    assert 0 <= row["minority_f1"] <= 1
