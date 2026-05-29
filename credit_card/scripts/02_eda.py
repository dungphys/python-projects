"""
Script 02 — Exploratory Data Analysis: Churn Analysis
"""
# ─ Import libraries ─────────────────────────────────────────────────────────────────────── #
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os

# ─ Set up ───────────────────────────────────────────────────────────────────────────────── #
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

RED      = "#DC2626"
ORANGE   = "#F5851C"
GREEN    = "#17EB10"
LBLUE    = "#19EDF5"
BLUE     = "#2563EB"
PURPLE   = "#EC4899"
GRAY     = "#64748B"
LIGHT    = "#F1F5F9"
PALETTE  = [RED, ORANGE, GREEN, LBLUE, BLUE, PURPLE, GRAY, LIGHT]

plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : LIGHT,
    "axes.edgecolor"   : "#CBD5E1",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "white",
    "grid.linewidth"   : 1.0,
    "font.family"      : "sans-serif",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
})

churn_colors = {0: BLUE, 1: RED}

# ────────────────────────────────────────────────────────────────────────────────────────── #
def load_data(data_path: str) -> pd.DataFrame:
    """
    Importing data and basic statistics
    """
    df = pd.read_csv(data_path)
    print("Data info:\n", df.info())
    print("Categorical description:\n")
    print(df.describe(include="object").T)
    print("\n\nNumerical description:\n")
    print(df.describe(include=np.number).T)
    print("\n\nData loaded successfully!\n")
    
    return df

# ────────────────────────────────────────────────────────────────────────────────────────── #
def churn_by_cat(
    df: pd.DataFrame,
    target_col: str
) -> None:
    print("=" * 60)
    print("CREDIT CARD CHURN — EDA SUMMARY")
    print("=" * 60)
    print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    target_values = list(df[target_col].value_counts().index)
    target_counts = list(df[target_col].value_counts())
    for i, val in enumerate(target_values):
        print(f"{target_values[i]}: {100*target_counts[i]/np.sum(target_counts):.2f}%"
              f" ({target_counts[i]}/{np.sum(target_counts)})")
    # ─ Figure 1:
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols.remove(target_col)
    n_plots = len(cat_cols) + 1
    cols = 3
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("CREDIT CARD CHURN BY CATEGORICAL COLUMNS", fontsize=16, fontweight="bold", y=1.01)
    # -- Churn pie
    counts = df[target_col].value_counts()
    axes[0].pie(counts, labels=counts.index, colors=[BLUE, RED],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2})
    axes[0].set_title("Churn Overall Distribution")
    # -- Churn distribution by categorical features
    for i, col in enumerate(cat_cols):
        print(col)
        ct = pd.crosstab(df[col], df[target_col], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=axes[i+1], 
                color=[BLUE, RED], edgecolor="white")
        axes[i+1].set_title(f"Churn by {col}")
    plt.tight_layout()
    plt.savefig("../outputs/01_eda_churn_by_cat.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✔ Saved: outputs/01_eda_churn_by_cat.png")

# ────────────────────────────────────────────────────────────────────────────────────────── #
def churn_by_num(
    df: pd.DataFrame, 
    target_col: str 
) -> None:
    # ─ Figure 2:
    num_cols = df.select_dtypes(include='number').columns.tolist()
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink":0.8}, ax=ax, annot_kws={"size":8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("../outputs/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved: outputs/02_correlation_heatmap.png")

    # ─ Figure 3:
    n_plots = len(num_cols)
    cols = 3
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("CREDIT CARD CHURN BY NUMERICAL COLUMNS", fontsize=16, fontweight="bold", y=1.01)
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue=target_col, ax=axes[i])
    plt.tight_layout()
    plt.savefig("../outputs/02_eda_churn_by_num.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved: outputs/02_eda_churn_by_num.png")

# ────────────────────────────────────────────────────────────────────────────────────────── #
def run(data_path, target_col):
    print("[EDA] Loading Cleaned Data...")
    df = load_data(data_path)
    print("[EDA] Univariate Churn Analysis...")
    churn_by_cat(df, target_col)
    churn_by_num(df, target_col)
    print("[EDA] Bivariate Churn Analysis...")
    print("[EDA] Complete. All figures saved to outputs/")
 
 
if __name__ == "__main__":
    data_path = "../data/cleaned_data.csv" 
    target_col = "Attrition_Flag"
    run(data_path, target_col)
