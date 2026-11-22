# BBB Peptide Permeability — ML Screening Pipeline

Reproducible research-oriented peptide classification and candidate-prioritization tools for predicting blood–brain barrier (BBB) permeability **from amino-acid sequences**. The private labeled training dataset and trained model are **not included** in this public repository.

This is computational screening software, **not experimental evidence of transport**, a clinical device, or a validated drug-discovery decision system.

## What is implemented

- Deterministic 428-feature descriptor pipeline: 20 amino-acid composition, 400 dipeptide frequencies, 8 physicochemical/length descriptors.
- Robust peptide validation: noncanonical residues raise errors rather than silently changing the sequence.
- Reproducible Logistic Regression or XGBoost training with sequence deduplication, conflicting-label removal, class-imbalance handling, and stratified train/validation/test partitions.
- Validation-only MCC threshold selection, independently evaluated test split, both-class PR-AUC, ROC-AUC, confusion matrix, balanced accuracy and Brier score.
- Local model bundle with consistent feature columns and threshold; batch inference and single-amino-acid substitution ranking.
- Two-tab Gradio interface and offline automated tests / GitHub Actions CI.

## Run locally

Python 3.11 recommended. From the repository root:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

To **train a new model** with your own appropriately licensed/private data, create a local CSV (do not commit it):

```csv
sequence,label
DSSHAFTLDELR,1
ACDEFGHIKLMN,0
```

These are **format examples only**, not verified experimental permeability labels or a usable training dataset. Training requires at least 20 unique sequences per class, and meaningful evaluation requires substantially more and independent evidence.

```bash
python -m src.train_model --data /absolute/path/to/private_peptides.csv --model xgboost
# Faster baseline:
python -m src.train_model --data /absolute/path/to/private_peptides.csv --model logistic
```

This writes `models/final_bbb_model.pkl` and `models/metrics.json` **locally**. Metrics are produced by running the training code, not reported as already measured results.

```bash
python -m src.app.gradio_app
python -m src.app.main
```

The UI has **Batch prediction** (up to 500 sequences) and **Single-residue scan** (ranks individual amino-acid substitutions relative to the parent). Inference will display a clear missing-model error until you train locally or supply your own compatible authorized model. Only load trusted `.pkl` / joblib artifacts, since loading them can execute code.

## Research integrity / limitations

- The historical project README mentioned an XGBoost ROC-AUC near **0.70** and positive-class PR-AUC near **0.92**. These are **previously reported approximate results, not independently reproduced by this repository**. This upgrade does not claim improved accuracy without evaluating the private dataset.
- Exact duplicate sequences and contradictory labels are removed **before splitting** to prevent exact-sequence train/test leakage. This is **not** a homology/sequence-cluster split; related peptides may still span partitions. Evaluate an additional sequence-similarity group split before making generalization claims.
- PR-AUC must always be presented with the **positive class and class prevalence** because extreme imbalance can make it appear high by chance. Both BBB+ and BBB− PR-AUCs are reported.
- XGBoost prediction outputs are **model scores, not guaranteed calibrated probabilities**. A Brier score is calculated for diagnostic comparison, but does not establish calibration. Classification thresholds are selected **only on validation data**.
- BBB transport can depend on peptide modifications, cargo, route, assay and biological context not captured by these sequence descriptors. In-silico substitutions are hypotheses, not measured gains.
- Included `data/sample/` files contain **small-molecule BBB datasets and heterogeneous peptide study records** for historical exploratory analysis; they are **not** a clean labeled dataset appropriate for this sequence-only model. Do not relabel small-molecule SMILES as peptides.
- The root-level `features.py`, `modeling.py`, `agent_tool.py`, and `nlp.py` reflect a separate historical 12-residue stacking experiment. They are not interchangeable with the 428-column `src/` model artifact and are not invoked by the new training/UI pipeline.

## Source layout

| Path | Purpose |
| --- | --- |
| `src/features.py` | Deterministic peptide validation and feature engineering |
| `src/train_model.py` | Private-data training, validation and model export |
| `src/predictor.py` | Consistent artifact loading and batch inference |
| `src/optimizer.py` | Single-substitution ranking |
| `src/app/gradio_app.py` | Local interactive screening UI |
| `tests/test_peptide_pipeline.py` | Synthetic offline regression and end-to-end checks |
| `.github/workflows/python-package-conda.yml` | Automated Python checks |

## Privacy

Training sequences, proprietary labels and model artifacts stay local. The `.gitignore` excludes `private_data/`, `models/`, `.env`, `*.pkl`, and generated output. Check repository history and the dataset license before sharing additional assets. Public sample records are illustrative and should not be treated as confidential training labels.
