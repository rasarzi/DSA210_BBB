import pandas as pd

# Load datasets
b3db = pd.read_csv("B3DB_classification.csv")
bbbp = pd.read_csv("BBBP.csv")
proj = pd.read_csv("proj201database.xlsx - Sayfa1 (1).csv")

# Clean column names
b3db.columns = [c.strip() for c in b3db.columns]
bbbp.columns = [c.strip() for c in bbbp.columns]
proj.columns = [c.strip() for c in proj.columns]

print("\n=== DATASET SHAPES ===")
print("B3DB:", b3db.shape)
print("BBBP:", bbbp.shape)
print("PROJ201:", proj.shape)

print("\n=== B3DB COLUMNS ===")
print(b3db.columns.tolist())

print("\n=== BBBP COLUMNS ===")
print(bbbp.columns.tolist())

print("\n=== PROJ201 COLUMNS ===")
print(proj.columns.tolist())

# B3DB label distribution
print("\n=== B3DB LABEL DISTRIBUTION ===")
print(b3db["BBB+/BBB-"].value_counts(dropna=False))

# BBBP label distribution
print("\n=== BBBP LABEL DISTRIBUTION ===")
print(bbbp["p_np"].value_counts(dropna=False))

# Standardize only for conflict checking between B3DB and BBBP
b3db_small = b3db[["SMILES", "BBB+/BBB-"]].copy()
b3db_small = b3db_small.rename(columns={"BBB+/BBB-": "BBB"})

bbbp_small = bbbp[["smiles", "p_np"]].copy()
bbbp_small = bbbp_small.rename(columns={"smiles": "SMILES", "p_np": "BBB"})

# Merge B3DB + BBBP for label conflict analysis
merged = pd.concat([b3db_small, bbbp_small], ignore_index=True)

conflicts = merged.groupby("SMILES")["BBB"].nunique()
conflicts = conflicts[conflicts > 1]

print("\n=== CROSS-DATASET LABEL CONFLICTS ===")
print(f"Conflicting molecules: {len(conflicts)}")

# PROJ201 insight
print("\n=== PROJ201 OUTCOME TYPE DISTRIBUTION ===")
print(proj["Outcome_Type"].value_counts(dropna=False).head(10))

print("\n=== PROJ201 LABEL CONFIDENCE DISTRIBUTION ===")
print(proj["Label_Confidence"].value_counts(dropna=False).head(10))

b3db.head(100).to_csv("b3db_sample.csv", index=False)
bbbp.head(100).to_csv("bbbp_sample.csv", index=False)
proj.head(100).to_csv("proj201_sample.csv", index=False)