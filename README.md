# BBB Dataset Cleaning & Analysis

This project focuses on improving dataset quality for Blood-Brain Barrier (BBB) permeability prediction.

## What is done
- Merged multiple datasets (B3DB, BBBP, LogBB)
- Standardized SMILES and labels
- Removed duplicates
- Identified label inconsistencies
- Analyzed class imbalance

## Goal
Improve reliability of negative samples (BBB−) using filtering and uncertainty-aware approaches.

## Run
```bash
DSA210_BBB/
│
├── data/
│   ├── bbbp_sample.csv
│   ├── proj201_sample(1).csv
│   └── b3db_sample.csv
│
├── outputs/   ← optional (can ignore or keep)
│
├── src/
│   ├── load_data.py
│   ├── clean_data.py
│   └── analyze_data.py
│
├── milestone_analysis.ipynb
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
