import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, spearmanr

def run_analysis(data, output_dir):
    bbbp = data["bbbp"]
    b3db = data["b3db"]
    proj = data["proj201"]

    # 📊 EDA
    plt.figure()
    bbbp["bbb_label"].value_counts().plot(kind="bar")
    plt.title("BBB Label Distribution")
    plt.savefig(output_dir / "class_distribution.png")
    plt.close()

    plt.figure()
    bbbp.boxplot(column="smiles_length", by="bbb_label")
    plt.savefig(output_dir / "smiles_length_boxplot.png")
    plt.close()

    # 🧪 Hypothesis Test 1
    group1 = bbbp[bbbp["bbb_label"] == 1]["smiles_length"]
    group2 = bbbp[bbbp["bbb_label"] == 0]["smiles_length"]

    t_stat, p_val = ttest_ind(group1, group2)
    print("T-test:", t_stat, p_val)

    # 🧪 Hypothesis Test 2
    corr, p = spearmanr(b3db["smiles_length"], b3db["logBB"])
    print("Spearman:", corr, p)

    # 🧪 Hypothesis Test 3
    g1 = proj[proj["has_modification"] == 1]["Length_num"]
    g2 = proj[proj["has_modification"] == 0]["Length_num"]

    t_stat2, p_val2 = ttest_ind(g1, g2)
    print("Modification test:", t_stat2, p_val2)S (sample) ===")
print(proj["Outcome_Value"].value_counts().head(10))
