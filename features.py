import math
import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Standard amino acids used for explicit one-hot features.
AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AAS)

# Physicochemical lookup tables.
# Unknown/ambiguous residues are mapped to neutral/default values.
HYDRO = dict(
    A=1.8, C=2.5, D=-3.5, E=-3.5, F=2.8, G=-0.4, H=-3.2, I=4.5,
    K=-3.9, L=3.8, M=1.9, N=-3.5, P=-1.6, Q=-3.5, R=-4.5, S=-0.8,
    T=-0.7, V=4.2, W=-0.9, Y=-1.3, X=0.0
)

CHARGE = dict(
    K=1.0, R=1.0, H=0.5, D=-1.0, E=-1.0,
    A=0.0, C=0.0, F=0.0, G=0.0, I=0.0, L=0.0, M=0.0, N=0.0,
    P=0.0, Q=0.0, S=0.0, T=0.0, V=0.0, W=0.0, Y=0.0, X=0.0
)

VOLUME = dict(
    A=88.6, C=108.5, D=111.1, E=138.4, F=189.9, G=60.1, H=153.2,
    I=166.7, K=168.6, L=166.7, M=162.9, N=114.1, P=112.7, Q=143.8,
    R=173.4, S=89.0, T=116.1, V=140.0, W=227.8, Y=193.6, X=130.0
)

VALID_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYX]+$")


def clean_sequence(seq: str, expected_length: int = 12) -> str:
    """Normalize a peptide sequence.

    Keeps only uppercase alphabetic amino-acid-like characters.
    Raises ValueError if the sequence is invalid or has the wrong length.
    """
    if seq is None:
        raise ValueError("Sequence is None")

    seq = re.sub(r"[^A-Za-z]", "", str(seq)).upper()

    if len(seq) != expected_length:
        raise ValueError(f"Expected length {expected_length}, got {len(seq)} for sequence {seq!r}")

    if not VALID_SEQUENCE_RE.match(seq):
        raise ValueError(f"Invalid amino-acid characters in sequence {seq!r}")

    return seq


def try_clean_sequence(seq: str, expected_length: int = 12) -> str | None:
    try:
        return clean_sequence(seq, expected_length=expected_length)
    except ValueError:
        return None


def featurize_one_sequence(seq: str, expected_length: int = 12) -> np.ndarray:
    """302-feature representation for a 12-aa peptide.

    Features:
    - 20 amino acid composition
    - 12 x 20 position one-hot
    - 12 hydrophobicity
    - 12 charge
    - 12 volume
    - 6 global summaries
    """
    seq = clean_sequence(seq, expected_length=expected_length)
    L = len(seq)

    # 1) Composition
    comp = [seq.count(aa) / L for aa in AAS]

    # 2) Position one-hot
    pos_oh = []
    for i in range(expected_length):
        residue = seq[i]
        for aa in AAS:
            pos_oh.append(1.0 if residue == aa else 0.0)

    # 3) Per-position physicochemical
    hydro_p = [HYDRO.get(seq[i], 0.0) / 5.0 for i in range(expected_length)]
    charge_p = [CHARGE.get(seq[i], 0.0) for i in range(expected_length)]
    volume_p = [VOLUME.get(seq[i], 130.0) / 230.0 for i in range(expected_length)]

    # 4) Global summaries
    total_h = sum(HYDRO.get(c, 0.0) for c in seq) / (5.0 * L)
    total_charge = sum(CHARGE.get(c, 0.0) for c in seq) / L
    frac_pos_charge = sum(1 for c in seq if CHARGE.get(c, 0.0) > 0) / L
    frac_neg_charge = sum(1 for c in seq if CHARGE.get(c, 0.0) < 0) / L
    frac_aromatic = sum(1 for c in seq if c in "FYW") / L
    frac_high_information = sum(1 for c in seq if c in "DEGHKPQRWY") / L

    glob = [
        total_h,
        total_charge,
        frac_pos_charge,
        frac_neg_charge,
        frac_aromatic,
        frac_high_information,
    ]

    return np.array(comp + pos_oh + hydro_p + charge_p + volume_p + glob, dtype=np.float32)


def featurize_sequences(sequences: Sequence[str], expected_length: int = 12) -> np.ndarray:
    return np.vstack([featurize_one_sequence(s, expected_length=expected_length) for s in sequences]).astype(np.float32)


def feature_names(expected_length: int = 12) -> List[str]:
    names = []
    names += [f"comp_{aa}" for aa in AAS]
    names += [f"pos{i+1}_{aa}" for i in range(expected_length) for aa in AAS]
    names += [f"hydro_pos{i+1}" for i in range(expected_length)]
    names += [f"charge_pos{i+1}" for i in range(expected_length)]
    names += [f"volume_pos{i+1}" for i in range(expected_length)]
    names += [
        "global_hydrophobicity",
        "global_charge",
        "frac_positive_charge",
        "frac_negative_charge",
        "frac_aromatic",
        "frac_high_information_residues",
    ]
    return names
