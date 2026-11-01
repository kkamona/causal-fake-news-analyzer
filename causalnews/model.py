"""Training, evaluation and persistence of the fake-news classifier."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_articles.csv"
SEED = 42


def default_model_path() -> Path:
    env = os.environ.get("CAUSALNEWS_MODEL")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models" / "model.joblib"


def load_dataset(path: Path = DATA_PATH) -> tuple[list[str], list[int]]:
    """Return ``(texts, labels)`` where label 1 means *fake*."""
    texts: list[str] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            texts.append(row["text"])
            labels.append(1 if row["label"] == "fake" else 0)
    return texts, labels


def _pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=SEED)),
    ])


def vectorise(feature_dicts: list[dict[str, float]]) -> np.ndarray:
    return np.array([[f[name] for name in FEATURE_NAMES] for f in feature_dicts])


def _dataset_features() -> tuple[np.ndarray, np.ndarray]:
    from .analyzer import compute_features  # local import avoids a cycle

    texts, labels = load_dataset()
    return vectorise([compute_features(t) for t in texts]), np.array(labels)


def train(path: Path | None = None) -> Pipeline:
    """Fit the model on the bundled corpus and optionally save it."""
    x, y = _dataset_features()
    model = _pipeline().fit(x, y)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
    return model


def load_or_train(path: Path | None = None) -> Pipeline:
    path = path or default_model_path()
    if path.exists():
        return joblib.load(path)
    return train(path)


def cross_validate(folds: int = 5) -> dict[str, float]:
    """Stratified k-fold evaluation on the bundled corpus."""
    x, y = _dataset_features()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(_pipeline(), x, y, cv=splitter)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted)),
        "recall": float(recall_score(y, predicted)),
        "f1": float(f1_score(y, predicted)),
        "samples": int(len(y)),
        "folds": folds,
    }
