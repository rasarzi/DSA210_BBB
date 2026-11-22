"""Deterministic 428-dimensional sequence descriptors for BBB peptide screening."""

from itertools import product
import re

import numpy as np
import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = [a + b for a, b in product(AMINO_ACIDS, repeat=2)]
HYDROPHOBIC = set("AVILMFWY")
AROMATIC = set("FWY")
POLAR = set("STNQCY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDRO_SCORES = dict(zip(AMINO_ACIDS, [
    1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
    1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3,
]))
FEATURE_COLUMNS = (
    [f"aac_{aa}" for aa in AMINO_ACIDS]
    + [f"dipep_{pair}" for pair in DIPEPTIDES]
    + ["length", "hydrophobic", "aromatic", "polar", "positive", "negative", "net_charge", "gravy"]
)


def clean_sequence(sequence: str, min_length: int = 2, max_length: int = 100) -> str:
    """Normalize space/hyphen-delimited peptide notation without silently deleting residues."""
    if not isinstance(sequence, str):
        raise ValueError("Peptide sequence must be a string.")
    seq = re.sub(r"[\s-]+", "", sequence).upper()
    if not min_length <= len(seq) <= max_length:
        raise ValueError(f"Peptide length must be between {min_length} and {max_length} residues.")
    invalid = sorted(set(seq) - set(AMINO_ACIDS))
    if invalid:
        raise ValueError(f"Noncanonical residues: {', '.join(invalid)}. Use the 20 standard amino acids.")
    return seq


def _vector(sequence: str) -> np.ndarray:
    seq = clean_sequence(sequence)
    length = len(seq)
    counts = {aa: seq.count(aa) for aa in AMINO_ACIDS}
    aac = [counts[aa] / length for aa in AMINO_ACIDS]
    pairs = {pair: 0 for pair in DIPEPTIDES}
    for i in range(length - 1):
        pairs[seq[i:i + 2]] += 1
    dipeps = [pairs[pair] / (length - 1) for pair in DIPEPTIDES]
    phys = [
        length,
        sum(counts[aa] for aa in HYDROPHOBIC) / length,
        sum(counts[aa] for aa in AROMATIC) / length,
        sum(counts[aa] for aa in POLAR) / length,
        sum(counts[aa] for aa in POSITIVE) / length,
        sum(counts[aa] for aa in NEGATIVE) / length,
        (counts["K"] + counts["R"] + counts["H"] - counts["D"] - counts["E"]) / length,
        sum(HYDRO_SCORES[aa] for aa in seq) / length,
    ]
    return np.asarray(aac + dipeps + phys, dtype=np.float32)


def make_features(sequence: str) -> pd.DataFrame:
    return pd.DataFrame([_vector(sequence)], columns=FEATURE_COLUMNS)


def make_feature_matrix(sequences) -> pd.DataFrame:
    vectors = [_vector(sequence) for sequence in sequences]
    return pd.DataFrame(np.vstack(vectors), columns=FEATURE_COLUMNS) if vectors else pd.DataFrame(columns=FEATURE_COLUMNS)
