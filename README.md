# BBB Peptide Classifier & Dataset Analysis

This project focuses on building a sequence-based machine learning classifier for predicting Blood-Brain Barrier (BBB) permeability of peptide sequences.

The training dataset is private and is not included in this repository.

## What is done

- Loaded positive and negative BBB peptide datasets
- Cleaned and standardized peptide sequences
- Removed duplicate sequences
- Checked class imbalance and sequence validity
- Built sequence-based features:
  - Amino Acid Composition (AAC)
  - Dipeptide Composition
  - Physicochemical descriptors
- Trained baseline models:
  - Logistic Regression
  - XGBoost
- Tuned XGBoost using randomized hyperparameter search
- Built a reusable BBB prediction function
- Created a simple Gradio demo interface

## Current best model

Tuned XGBoost using AAC + dipeptide + physicochemical features.

Approximate validation performance:

| Model | Features | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| XGBoost | AAC + Dipeptide + Physicochemical | ~0.70 | ~0.92 |

## Goal

Develop a practical computational screening tool for ranking peptide sequences by predicted BBB permeability.

The model is intended for candidate prioritization, not experimental validation.

## Repository structure

```text
DSA210_BBB/
│
├── data/
│   └── sample_sequences.csv
│
├── outputs/
│   └── predictions.csv
│
├── models/
│   ├── final_bbb_model.pkl          # optional, if allowed
│   └── feature_columns.pkl          # optional, if allowed
│
├── src/
│   ├── features.py
│   ├── train_model.py
│   └── predictor.py
│
├── app/
│   └── gradio_app.py
│
├── notebooks/
│   └── milestone_analysis.ipynb
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
