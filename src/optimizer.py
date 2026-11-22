"""Single-residue in-silico substitution scan; not an experimental optimization claim."""

from src.features import AMINO_ACIDS, clean_sequence


def scan_single_mutations(sequence: str, predictor, limit: int = 20):
    seq = clean_sequence(sequence)
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    variants = []
    info = []
    for index, original in enumerate(seq):
        for replacement in AMINO_ACIDS:
            if replacement == original:
                continue
            variants.append(seq[:index] + replacement + seq[index + 1:])
            info.append((index + 1, original, replacement))
    baseline = predictor.predict_many([seq])[0]["bbb_score"]
    predictions = predictor.predict_many(variants)
    rows = []
    for (pos, old, new), prediction in zip(info, predictions):
        rows.append({
            "position": pos,
            "substitution": f"{old}{pos}{new}",
            "sequence": prediction["sequence"],
            "bbb_score": prediction["bbb_score"],
            "delta_from_parent": round(prediction["bbb_score"] - baseline, 6),
        })
    rows.sort(key=lambda r: (-r["delta_from_parent"], r["position"], r["substitution"]))
    return {"original_sequence": seq, "original_score": baseline, "top_variants": rows[:limit], "total_variants": len(rows)}
