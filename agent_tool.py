from typing import Dict, List

import numpy as np

from .features import featurize_sequences, clean_sequence
from .modeling import (
    load_bundle,
    make_meta_features,
    confidence_from_probability,
    predict_proba_positive,
)
from .nlp import extract_sequences_from_prompt


def predict_sequences(sequences: List[str], model_path: str) -> List[Dict]:
    """Predict BBB permeability for explicit peptide sequences."""
    bundle = load_bundle(model_path)
    expected_length = bundle.get("expected_length", 12)
    threshold = float(bundle["chosen_threshold"])

    cleaned = [clean_sequence(s, expected_length=expected_length) for s in sequences]
    X_rich = featurize_sequences(cleaned, expected_length=expected_length)

    p_main = predict_proba_positive(bundle["main_model"], X_rich)
    p_kmer = predict_proba_positive(bundle["kmer_model"], cleaned)

    # Some bundles may not include a separate meta model.
    if "meta_model" in bundle and bundle["meta_model"] is not None:
        X_meta = make_meta_features(X_rich, p_main, p_kmer)
        p_pos = predict_proba_positive(bundle["meta_model"], X_meta)
        model_used = "stacked"
    else:
        p_pos = p_main
        model_used = "main"

    results = []
    for seq, p in zip(cleaned, p_pos):
        p = float(p)
        classification = "BBB+" if p >= threshold else "BBB-"
        results.append({
            "sequence": seq,
            "p_bbb_positive": round(p, 6),
            "p_bbb_negative": round(1.0 - p, 6),
            "classification": classification,
            "threshold_used": round(threshold, 6),
            "confidence": confidence_from_probability(p, threshold),
            "model_used": model_used,
            "warning": "Prediction only; experimental validation is required."
        })
    return results


def predict_bbb_from_prompt(prompt: str, model_path: str) -> Dict:
    """Agent-callable tool.

    The AI agent can pass the scientist's natural-language prompt here.
    The function extracts 12-aa peptide sequences and returns predictions.
    """
    bundle = load_bundle(model_path)
    expected_length = bundle.get("expected_length", 12)

    sequences = extract_sequences_from_prompt(prompt, expected_length=expected_length)
    if not sequences:
        return {
            "status": "no_sequence_found",
            "message": f"No valid {expected_length}-amino-acid peptide sequence found in the prompt.",
            "predictions": []
        }

    return {
        "status": "ok",
        "n_sequences": len(sequences),
        "predictions": predict_sequences(sequences, model_path=model_path)
    }
