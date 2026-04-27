import itertools
import pandas as pd

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDES = ["".join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]

HYDROPHOBIC = set("AVILMFWY")
AROMATIC = set("FWY")
POLAR = set("STNQCY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")

HYDRO_SCORES = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper().replace("X", "")
    seq = "".join([aa for aa in seq if aa in AMINO_ACIDS])
    if len(seq) < 2:
        raise ValueError("Sequence must contain at least 2 valid amino acids.")
    return seq

def make_features(seq: str) -> pd.DataFrame:
    seq = clean_sequence(seq)
    L = len(seq)

    aac = [seq.count(a) / L for a in AMINO_ACIDS]

    total = L - 1
    dp_counts = {dp: 0 for dp in DIPEPTIDES}
    for i in range(total):
        dp = seq[i:i+2]
        if dp in dp_counts:
            dp_counts[dp] += 1
    dp_features = [dp_counts[dp] / total for dp in DIPEPTIDES]

    c = {a: seq.count(a) for a in AMINO_ACIDS}
    phys = [
        L,
        sum(c[a] for a in HYDROPHOBIC) / L,
        sum(c[a] for a in AROMATIC) / L,
        sum(c[a] for a in POLAR) / L,
        sum(c[a] for a in POSITIVE) / L,
        sum(c[a] for a in NEGATIVE) / L,
        (c["K"] + c["R"] + c["H"] - c["D"] - c["E"]) / L,
        sum(HYDRO_SCORES.get(a, 0) for a in seq) / L,
    ]

    columns = (
        AMINO_ACIDS
        + DIPEPTIDES
        + [
            "length",
            "hydrophobic",
            "aromatic",
            "polar",
            "positive",
            "negative",
            "net_charge",
            "gravy",
        ]
    )

    return pd.DataFrame([aac + dp_features + phys], columns=columns)
