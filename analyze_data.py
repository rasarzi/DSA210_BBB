import pandas as pd

# --- LOAD DATASETS ---
b3db = pd.read_csv("B3DB_classification.csv")
bbbp = pd.read_csv("BBBP.csv")
proj = pd.read_csv("proj201database.xlsx - Sayfa1 (1).csv")

# --- STANDARDIZE COLUMNS ---
b3db.columns = [c.strip() for c in b3db.columns]
bbbp.columns = [c.strip() for c in bbbp.columns]
proj.columns = [c.strip() for c in proj.columns]

bbbp = bbbp.rename(columns={"smiles": "SMILES", "p_np": "BBB"})

# --- BASIC INFO ---
print("\n=== DATASET SHAPES ===")
print("B3DB:", b3db.shape)
print("BBBP:", bbbp.shape)
print("PROJ201:", proj.shape)

print("\n=== B3DB CLASS DISTRIBUTION ===")
print(b3db["BBB"].value_counts())

print("\n=== BBBP CLASS DISTRIBUTION ===")
print(bbbp["BBB"].value_counts())

# --- CHECK LABEL CONFLICTS BETWEEN DATASETS ---
merged = pd.concat([
    b3db[["SMILES", "BBB"]],
    bbbp[["SMILES", "BBB"]]
], ignore_index=True)

conflicts = merged.groupby("SMILES")["BBB"].nunique()
conflicts = conflicts[conflicts > 1]

print("\n=== CROSS-DATASET LABEL CONFLICTS ===")
print(f"Conflicting molecules: {len(conflicts)}")

# --- PROJ201 QUICK INSIGHT ---
print("\n=== PROJ201 OUTCOME TYPES ===")
print(proj["Outcome_Type"].value_counts().head(10))

print("\n=== PROJ201 OUTCOME VALUES (sample) ===")
print(proj["Outcome_Value"].value_counts().head(10))