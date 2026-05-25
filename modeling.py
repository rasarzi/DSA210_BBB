import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

from .features import featurize_sequences


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_kmer_pipeline() -> Pipeline:
    """Auxiliary k-mer model. Good for motif-like signal and BBB− recall support."""
    return Pipeline([
        ("vectorizer", CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False, binary=False)),
        ("clf", ComplementNB(alpha=0.5)),
    ])


def build_main_model(random_state: int = 42):
    """Main rich-feature model.

    LightGBM is preferred. If LightGBM is unavailable, falls back to calibrated SGD.
    """
    if HAS_LIGHTGBM:
        return LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_data_in_leaf=100,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_lambda=5.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    # fallback: fast and stable
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=random_state,
        )),
    ])


def build_meta_model(random_state: int = 42):
    """Meta model on [rich features + base probabilities].

    LightGBM can be used here too, but a calibrated linear model is faster and safer
    for deployment.
    """
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            alpha=1e-4,
            max_iter=3000,
            tol=1e-3,
            random_state=random_state,
        )),
    ])


def predict_proba_positive(model, X):
    probs = model.predict_proba(X)
    return probs[:, 1]


def make_meta_features(X_rich: np.ndarray, p_main: np.ndarray, p_kmer: np.ndarray) -> np.ndarray:
    """Combine rich peptide features and model-level signals."""
    return np.column_stack([
        X_rich,
        p_main,
        safe_logit(p_main),
        p_kmer,
        safe_logit(p_kmer),
    ]).astype(np.float32)


def compute_metrics(y_true, p_pos, threshold: float) -> Dict[str, float]:
    y_pred = (p_pos >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, p_pos)),
        "pr_auc_bbb_negative": float(average_precision_score((y_true == 0).astype(int), 1.0 - p_pos)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "bbb_negative_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "bbb_negative_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "bbb_negative_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "bbb_positive_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "bbb_positive_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "bbb_positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn_true_neg_pred_neg": int(tn),
        "fp_true_neg_pred_pos": int(fp),
        "fn_true_pos_pred_neg": int(fn),
        "tp_true_pos_pred_pos": int(tp),
    }
    return metrics


def choose_threshold(y_val, p_pos_val, mode: str = "mcc") -> Tuple[float, Dict[str, float]]:
    """Choose threshold using validation set only.

    mode options:
    - "mcc": best overall correlation
    - "balanced_accuracy": best mean recall
    - "bbb_negative_f2": emphasizes BBB− recall
    - "bbb_negative_recall_precision30": max BBB− recall with BBB− precision >= 0.30
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []

    for t in thresholds:
        m = compute_metrics(y_val, p_pos_val, threshold=float(t))

        precision_neg = m["bbb_negative_precision"]
        recall_neg = m["bbb_negative_recall"]
        beta2 = 4.0
        denom = beta2 * precision_neg + recall_neg
        f2_neg = (1.0 + beta2) * precision_neg * recall_neg / denom if denom > 0 else 0.0
        m["bbb_negative_f2"] = float(f2_neg)

        if mode == "mcc":
            score = m["mcc"]
        elif mode == "balanced_accuracy":
            score = m["balanced_accuracy"]
        elif mode == "bbb_negative_f2":
            score = m["bbb_negative_f2"]
        elif mode == "bbb_negative_recall_precision30":
            score = m["bbb_negative_recall"] if m["bbb_negative_precision"] >= 0.30 else -1.0
        else:
            raise ValueError(f"Unknown threshold mode: {mode}")

        m["threshold_score"] = float(score)
        rows.append(m)

    best = max(rows, key=lambda r: r["threshold_score"])
    return float(best["threshold"]), best


def confidence_from_probability(p_pos: float, threshold: float) -> str:
    margin = abs(p_pos - threshold)
    if margin >= 0.25:
        return "high"
    if margin >= 0.10:
        return "medium"
    return "low"


def save_bundle(bundle: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str) -> Dict:
    return joblib.load(path)
