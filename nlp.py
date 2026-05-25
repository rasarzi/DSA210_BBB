import re
from typing import List

from .features import try_clean_sequence

# Extract candidate peptides from natural language.
# For this project we expect 12-aa peptides.
TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z\-\s]{8,30}[A-Za-z]\b")


def extract_sequences_from_prompt(prompt: str, expected_length: int = 12) -> List[str]:
    """Extract valid peptide sequences from a natural-language prompt.

    Handles:
    - "RGDPRKGRGDSY"
    - "R G D P R K G R G D S Y"
    - "R-G-D-P-R-K-G-R-G-D-S-Y"

    It intentionally ignores invalid tokens and returns unique sequences in order.
    """
    if not prompt:
        return []

    candidates = []

    # First, catch clean uppercase/lowercase contiguous tokens.
    for tok in re.findall(r"\b[A-Za-z]{%d}\b" % expected_length, prompt):
        cleaned = try_clean_sequence(tok, expected_length=expected_length)
        if cleaned:
            candidates.append(cleaned)

    # Second, catch spaced or hyphen-separated peptide strings.
    for raw in TOKEN_RE.findall(prompt):
        compact = re.sub(r"[^A-Za-z]", "", raw)
        cleaned = try_clean_sequence(compact, expected_length=expected_length)
        if cleaned:
            candidates.append(cleaned)

    # Unique while preserving order.
    seen = set()
    out = []
    for seq in candidates:
        if seq not in seen:
            seen.add(seq)
            out.append(seq)
    return out
