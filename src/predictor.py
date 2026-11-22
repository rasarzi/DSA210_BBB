"""Inference for locally trained or explicitly supplied BBB peptide models."""

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.features import AMINO_ACIDS, DIPEPTIDES, FEATURE_COLUMNS, clean_sequence, make_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final_bbb_model.pkl"


class BBBPredictor:
    def __init__(self, model_path=MODEL_PATH):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No trained model at {path}. Run `python -m src.train_model --data /path/to/private_peptides.csv` "
                "or supply an authorized compatible model."
            )
        # joblib/pickle must ONLY be loaded from a source you trust.
        artifact = joblib.load(path)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.columns = list(artifact.get("feature_columns", FEATURE_COLUMNS))
            self.threshold = float(artifact.get("threshold", 0.5))
            self.metadata = artifact.get("metadata", {})
        else:
            self.model = artifact
            legacy_columns_path = path.with_name("feature_columns.pkl")
            self.columns = list(joblib.load(legacy_columns_path)) if legacy_columns_path.exists() else list(FEATURE_COLUMNS)
            self.threshold = 0.5
            self.metadata = {"legacy_artifact": True}
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Invalid model threshold.")
        legacy_aliases = {name: f"aac_{name}" for name in AMINO_ACIDS}
        legacy_aliases.update({name: f"dipep_{name}" for name in DIPEPTIDES})
        self.source_columns = [col if col in FEATURE_COLUMNS else legacy_aliases.get(col, col) for col in self.columns]
        missing = set(self.source_columns) - set(FEATURE_COLUMNS)
        if missing:
            raise ValueError(f"Model expects unsupported features: {sorted(missing)[:5]}")

    def predict_many(self, sequences):
        cleaned = [clean_sequence(s) for s in sequences]
        if not cleaned:
            return []
        X = make_feature_matrix(cleaned).loc[:, self.source_columns]
        X.columns = self.columns
        scores = self.model.predict_proba(X)[:, 1]
        rows = []
        for seq, score in zip(cleaned, scores):
            p = float(score)
            if not np.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ValueError("Model returned an invalid prediction score.")
            rows.append({
                "sequence": seq,
                "bbb_score": round(p, 6),
                "prediction": "BBB+" if p >= self.threshold else "BBB-",
                "threshold": round(self.threshold, 6),
            })
        return rows


@lru_cache(maxsize=4)
def get_predictor(model_path=None):
    return BBBPredictor(model_path or MODEL_PATH)


def predict_bbb(sequence: str):
    """Backwards-compatible (score, label) interface."""
    row = get_predictor().predict_many([sequence])[0]
    return row["bbb_score"], row["prediction"]
