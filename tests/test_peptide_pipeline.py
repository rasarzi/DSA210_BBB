from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, clean_sequence, make_features, make_feature_matrix
from src.optimizer import scan_single_mutations
from src.predictor import BBBPredictor
from src.train_model import load_labeled_sequences, train_model


def test_normalization_rejects_noncanonical_residues_instead_of_deleting_them():
    assert clean_sequence("a-c d") == "ACD"
    for item in ("AXCD", "ACD1", "", "AAAAAAA*", None):
        with pytest.raises(ValueError):
            clean_sequence(item)


def test_feature_schema_is_stable_and_finite():
    seq = "DSSHAFTLDELR"
    features = make_features(seq)
    assert features.shape == (1, 428)
    assert features.columns.tolist() == FEATURE_COLUMNS
    assert np.isfinite(features.to_numpy()).all()
    assert np.isclose(features[[f"aac_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"]].sum(axis=1).iloc[0], 1.0)
    assert make_feature_matrix([seq, seq]).shape == (2, 428)


def test_cleaning_drops_duplicate_and_conflicting_labels(tmp_path):
    p = tmp_path / "rows.csv"
    rows = ([{"sequence": "ACDEFGHIKLMN", "label": "BBB+"}] * 2
            + [{"sequence": "ACDEFGHIKLMN", "label": "BBB-"}]
            + [{"sequence": "AAAAA" + "C" * i + "D" * (10 - i), "label": 1} for i in range(11)]
            + [{"sequence": "DDDDD" + "E" * i + "K" * (10 - i), "label": 0} for i in range(11)])
    pd.DataFrame(rows).to_csv(p, index=False)
    data, stats = load_labeled_sequences(p)
    assert stats["conflicting_sequences"] == 1
    assert "ACDEFGHIKLMN" not in set(data["sequence"])
    assert not data["sequence"].duplicated().any()


def test_end_to_end_training_inference_and_mutation_scan(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for label, alphabet in ((1, list("KRSAC")), (0, list("VILFD"))):
        seqs = set()
        while len(seqs) < 40:
            seqs.add("".join(rng.choice(alphabet, 12)))
        rows.extend({"sequence": sequence, "label": label} for sequence in seqs)
    data = tmp_path / "private.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    model_folder = tmp_path / "models"
    report = train_model(data, model_folder, model_type="logistic", seed=42)
    assert report["split_sizes"] == {"train": 56, "validation": 12, "test": 12}
    assert 0 <= report["test"]["roc_auc"] <= 1
    assert (model_folder / "metrics.json").exists()
    predictor = BBBPredictor(model_folder / "final_bbb_model.pkl")
    res = predictor.predict_many(["DSSHAFTLDELR", "SSSSSSSSSS"])
    assert len(res) == 2 and all(0 <= item["bbb_score"] <= 1 for item in res)
    mutation = scan_single_mutations("DSSHAFTLDELR", predictor, limit=8)
    assert mutation["total_variants"] == 12 * 19
    assert len(mutation["top_variants"]) == 8
    assert mutation["top_variants"][0]["delta_from_parent"] >= mutation["top_variants"][-1]["delta_from_parent"]
