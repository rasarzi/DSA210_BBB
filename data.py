import os
from typing import Tuple

import pandas as pd

from .features import clean_sequence


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path!r}. Use .xlsx, .xls, .csv, or .txt")


def _detect_sequence_column(df: pd.DataFrame) -> str:
    # Prefer obvious names.
    for col in df.columns:
        name = str(col).lower()
        if any(key in name for key in ["sequence", "seq", "peptide", "lbr"]):
            return col

    # Otherwise find first column that mostly looks like 12-aa sequences.
    best_col = None
    best_count = -1
    for col in df.columns:
        count = 0
        sample = df[col].dropna().astype(str).head(1000)
        for val in sample:
            try:
                clean_sequence(val)
                count += 1
            except Exception:
                pass
        if count > best_count:
            best_count = count
            best_col = col

    if best_col is None or best_count == 0:
        raise ValueError("Could not detect a sequence column. Rename it to 'sequence'.")
    return best_col


def load_positive_negative_files(positive_path: str, negative_path: str) -> pd.DataFrame:
    pos = _read_table(positive_path)
    neg = _read_table(negative_path)

    pos_col = _detect_sequence_column(pos)
    neg_col = _detect_sequence_column(neg)

    pos_df = pd.DataFrame({"sequence": pos[pos_col].astype(str), "label": 1})
    neg_df = pd.DataFrame({"sequence": neg[neg_col].astype(str), "label": 0})

    df = pd.concat([pos_df, neg_df], ignore_index=True)

    cleaned = []
    labels = []
    bad_rows = 0

    for seq, label in zip(df["sequence"], df["label"]):
        try:
            cleaned.append(clean_sequence(seq))
            labels.append(int(label))
        except ValueError:
            bad_rows += 1

    out = pd.DataFrame({"sequence": cleaned, "label": labels})
    out = out.drop_duplicates(subset=["sequence", "label"]).reset_index(drop=True)

    # If exact same sequence appears with both labels, drop ambiguous rows.
    counts = out.groupby("sequence")["label"].nunique()
    ambiguous = set(counts[counts > 1].index)
    if ambiguous:
        out = out[~out["sequence"].isin(ambiguous)].reset_index(drop=True)

    print(f"Loaded {len(out):,} clean sequences.")
    print(f"Dropped invalid rows: {bad_rows:,}")
    print(f"Dropped ambiguous cross-label sequences: {len(ambiguous):,}")
    print(out["label"].value_counts().rename(index={1: "BBB+", 0: "BBB-"}))

    return out
