"""Reproducible training from an authorized private peptide-level CSV.

Expected columns: sequence,label. Labels must be 0/1 or BBB-/BBB+.
No private sequences are included in model reports or version control.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, clean_sequence, make_feature_matrix


LABELS = {"1": 1, "0": 0, "bbb+": 1, "bbb-": 0}


def load_labeled_sequences(csv_path):
    df = pd.read_csv(csv_path, usecols=["sequence", "label"], dtype="string")
    valid = []
    invalid_count = 0
    for row in df.itertuples(index=False, name=None):
        seq, raw_label = row
        try:
            label = LABELS.get(str(raw_label).strip().lower())
            if label is None:
                raise ValueError("Invalid label")
            valid.append((clean_sequence(seq), label))
        except (ValueError, TypeError):
            invalid_count += 1
    records = pd.DataFrame(valid, columns=["sequence", "label"])
    if records.empty:
        raise ValueError("No valid labeled peptide sequences were found.")
    label_count = records.groupby("sequence")["label"].nunique()
    ambiguous = set(label_count[label_count > 1].index)
    records = records.loc[~records["sequence"].isin(ambiguous)].drop_duplicates("sequence").reset_index(drop=True)
    counts = records["label"].value_counts()
    if len(counts) != 2 or counts.min() < 10:
        raise ValueError("Need at least 10 distinct, unambiguous sequences of each class.")
    return records, {"invalid_rows": invalid_count, "conflicting_sequences": len(ambiguous), "unique_sequences": len(records), "positive_sequences": int(counts[1]), "negative_sequences": int(counts[0])}


def build_estimator(name, random_state):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced", random_state=random_state))
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("Install xgboost from requirements.txt, or choose --model logistic") from exc
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=350,
            max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
            min_child_weight=3, reg_lambda=5.0, random_state=random_state,
            n_jobs=4, tree_method="hist",
        )
    raise ValueError("model must be xgboost or logistic")


def choose_threshold(y_true, scores):
    """Tune only on validation data using Matthews correlation coefficient."""
    thresholds = np.arange(0.05, 0.951, 0.01)
    results = [(matthews_corrcoef(y_true, scores >= t), t) for t in thresholds]
    return float(max(results, key=lambda pair: (pair[0], -abs(pair[1] - 0.5)))[1])


def evaluate(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)), "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc_bbb_positive": float(average_precision_score(y_true, scores)),
        "pr_auc_bbb_negative": float(average_precision_score(1 - y_true, 1 - scores)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "f1_bbb_positive": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "f1_bbb_negative": float(f1_score(y_true, predicted, pos_label=0, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "confusion_matrix_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_model(data_path, output_dir="models", model_type="xgboost", seed=42):
    data, cleaning = load_labeled_sequences(data_path)
    if data["label"].value_counts().min() < 20:
        raise ValueError("Training requires at least 20 unique sequences per class after cleaning.")
    # Splitting unique sequences before calculating any fitted preprocessing avoids exact-sequence leakage.
    train, heldout = train_test_split(data, test_size=0.30, stratify=data["label"], random_state=seed)
    validation, test = train_test_split(heldout, test_size=0.50, stratify=heldout["label"], random_state=seed)
    sets = [set(part["sequence"]) for part in (train, validation, test)]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Exact sequence overlap between splits")

    X_train = make_feature_matrix(train["sequence"])
    X_val = make_feature_matrix(validation["sequence"])
    X_test = make_feature_matrix(test["sequence"])
    estimator = build_estimator(model_type, seed)
    fit_kwargs = {}
    if model_type == "xgboost":
        n_pos = int((train["label"] == 1).sum())
        n_neg = int((train["label"] == 0).sum())
        fit_kwargs["sample_weight"] = np.where(train["label"].to_numpy() == 1, n_neg / n_pos, 1.0)
    estimator.fit(X_train, train["label"], **fit_kwargs)

    p_val = estimator.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(validation["label"], p_val)
    p_test = estimator.predict_proba(X_test)[:, 1]
    metrics = {
        "model": model_type, "seed": seed, "cleaning": cleaning,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_method": "stratified sequence-level split (exact duplicates removed; not a sequence-similarity cluster split)",
        "validation": evaluate(validation["label"], p_val, threshold),
        "test": evaluate(test["label"], p_test, threshold),
        "warning": "Model scores are not guaranteed calibrated; BBB crossing requires experimental verification.",
    }
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": estimator, "threshold": threshold, "feature_columns": list(FEATURE_COLUMNS),
        "metadata": {"model_type": model_type, "seed": seed, "feature_count": len(FEATURE_COLUMNS), "validation_method": metrics["split_method"]},
    }
    joblib.dump(bundle, folder / "final_bbb_model.pkl")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a BBB peptide classifier from private labeled data")
    parser.add_argument("--data", required=True, help="Private CSV with sequence,label columns")
    parser.add_argument("--output", default="models", help="Output directory for local model/metrics")
    parser.add_argument("--model", choices=["xgboost", "logistic"], default="xgboost")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train_model(args.data, args.output, args.model, args.seed), indent=2))


if __name__ == "__main__":
    main()
