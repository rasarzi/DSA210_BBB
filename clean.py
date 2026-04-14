import pandas as pd

def clean_all_datasets(datasets):
    bbbp = datasets["bbbp"].copy()
    b3db = datasets["b3db"].copy()
    proj201 = datasets["proj201"].copy()

    # BBBP
    bbbp["bbb_label"] = pd.to_numeric(bbbp["p_np"], errors="coerce")
    bbbp["smiles_length"] = bbbp["smiles"].astype(str).str.len()

    # B3DB
    b3db["bbb_label"] = b3db["BBB+/BBB-"].map({"BBB+": 1, "BBB-": 0})
    b3db["smiles_length"] = b3db["SMILES"].astype(str).str.len()
    b3db["logBB"] = pd.to_numeric(b3db["logBB"], errors="coerce")

    # PROJ201
    proj201["Length_num"] = pd.to_numeric(proj201["Length"], errors="coerce")
    proj201["has_modification"] = proj201["Chemical_Modifications"].notna().astype(int)

    return {
        "bbbp": bbbp.dropna(),
        "b3db": b3db.dropna(),
        "proj201": proj201.dropna()
    }
