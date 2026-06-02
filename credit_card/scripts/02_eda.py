"""
Exploratory Data Analysis: Churn Analysis
    1. Data loading and snapshot
    2. Basic statistics
"""
# ─ Import libraries ─────────────────────────────────────────────────────────────────────── #
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import seaborn as sns
import warnings, os, json
from datetime import datetime, timezone, timedelta
from scipy import stats
from pathlib import Path


# ─ Set up 
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

INPUT = "data/cleaned_data.csv"
LOG_PATH = "logs/02_eda_log.json"

RED      = "#DC2626"
ORANGE   = "#F17D17"
DORANGE   = "#9B4A03"
GREEN    = "#17EB10"
LBLUE    = "#0497AA"
BLUE     = "#2563EB"
PURPLE   = "#B906DD"
GRAY     = "#64748B"
LIGHT    = "#F1F5F9"
PALETTE  = [RED, ORANGE, DORANGE, GREEN, LBLUE, BLUE, PURPLE, GRAY, LIGHT]

plt.rcParams.update({
    "figure.facecolor" : "white",
    "axes.facecolor"   : "#FEFFFF",
    "axes.edgecolor"   : GRAY,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "white",
    "grid.linewidth"   : 0.3,
    "font.family"      : "sans-serif",
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 12,
})

# - Audit log
log = {
    "run_at"     : datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S %Z"),
    "input_path" : INPUT,
    "output_path": "outputs/",
    "steps"      : [],
}
# - record
def record(step, detail):
    entry = {
        "step": step, 
        "detail": detail
    }
    log["steps"].append(entry)
    tag = f"[{step}]"
    print(f" {tag:<30} {detail}")


def _resolve_input_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    print(base_dir)
    candidates = [
        base_dir / "cleaned_data.csv",
        base_dir.parent / "cleaned_data.csv",
        base_dir / "data" / "cleaned_data.csv",
        base_dir.parent / "data" / "cleaned_data.csv",
        Path("cleaned_data.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find cleaned_data.csv in the current project structure. "
        "Place the file in the data directory."
    )

# ------------------------------------------------------------------
# STEP 1 — LOAD AND SNAPSHOT
# ------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("=" * 65)
    print("STEP 1 : Load and Snapshot")
    print("=" * 65)

    file_path = _resolve_input_path()
    df = pd.read_csv(file_path)
    orig_shape = df.shape
    record("load", 
        f"Read {orig_shape[0]:,} rows x {orig_shape[1]} cols from {INPUT}")
    print(f"\nShape   : {df.shape}")
    print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"  Dtypes  :\n{df.dtypes.to_string()}\n")
 
    log["input_shape"]  = list(orig_shape)
    log["input_dtypes"] = df.dtypes.astype(str).to_dict()
    return df

# ------------------------------------------------------------------
# STEP 2 — ANALYSIS 1: BASIC STATISTICS
# ------------------------------------------------------------------
def get_basic_statistics(df: pd.DataFrame) -> list[dict]:
    print("=" * 65)
    print("STEP 2 : Basic Statistics")
    print("=" * 65)

    cat_stat_report = df.describe(include='object').to_dict()
    num_stat_report = df.describe(include=np.number).to_dict()

    record("basic_stats", 
           f"Categorical statistics\n: {cat_stat_report}\n\n"
           + f"Numerical statistics\n: {num_stat_report}\n\n"
    )
    return cat_stat_report, num_stat_report

# ------------------------------------------------------------------
# STEP 3 — ANALYSIS 2: NUMERICAL DISTRIBUTION
# ------------------------------------------------------------------
def plot_numeric_distribution(df, cols=None):
    print("=" * 65)
    print("STEP 3 : Numerical Distribution")
    print("=" * 65)
    if cols is None:
        cols = df.select_dtypes(include=['number']).columns
    ncol = 3
    nrow = math.ceil(len(cols) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 5*nrow))
    fig.suptitle("NUMERICAL DISTRIBUTION", 
                 fontsize=16, fontweight="bold", 
                 y=1.01, color=DORANGE)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(data=df, x=col, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        # Get the patches (bars) from the plot
        patches = [patch for patch in axes[i].patches if isinstance(patch, plt.matplotlib.patches.Rectangle)]

        # Apply gradient colormap
        cmap = cm.get_cmap('Spectral')
        norm = plt.Normalize(vmin=0, vmax=len(patches))

        for i, patch in enumerate(patches):
            patch.set_facecolor(cmap(norm(i)))
    plt.tight_layout()
    plt.savefig("outputs/01_eda_num_distrib.png", dpi=150, bbox_inches="tight")
    print("\n✔ Saved: outputs/01_eda_num_distrib.png")
    plt.show()

# ------------------------------------------------------------------
# STEP 4 — ANALYSIS 3: CATEGORICAL DISTRIBUTION
# ------------------------------------------------------------------
def plot_categorical_counts(df, cols=None):
    print("=" * 65)
    print("STEP 4 : Categorical distribution")
    print("=" * 65)
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns
    ncol = 3
    nrow = math.ceil(len(cols) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 5*nrow))
    fig.suptitle("CATEGORICAL DISTRIBUTION", 
                 fontsize=16, fontweight="bold", 
                 y=1.01, color=DORANGE)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.countplot(data=df, x=col,
                      order = df[col].value_counts().index, 
                      ax=axes[i], palette=PALETTE)
        axes[i].set_title(f'Count of {col}', fontweight="bold", fontsize=14)
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/02_eda_cat_distrib.png", dpi=150, bbox_inches="tight")
    print("\n✔ Saved: outputs/02_eda_cat_distrib.png")
    plt.show()
    
# ------------------------------------------------------------------
# STEP 5 — ANALYSIS 4: CHURN ANALYSIS BY CATEGORY
# ------------------------------------------------------------------
def churn_by_cat(
    df: pd.DataFrame,
    target_col: str
) -> None:
    print("=" * 65)
    print("STEP 5 : Churn by categorical variables")
    print("=" * 65)

    print(f"Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    target_values = list(df[target_col].value_counts().index)
    target_counts = list(df[target_col].value_counts())
    for i, val in enumerate(target_values):
        print(f"{target_values[i]}: {100*target_counts[i]/np.sum(target_counts):.2f}%"
              f" ({target_counts[i]}/{np.sum(target_counts)})")
    # ─ Figure 1a:
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols.remove(target_col)
    n_plots = len(cat_cols) + 1
    cols = 3
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("CREDIT CARD CHURN BY CATEGORICAL COLUMNS", fontsize=16, fontweight="bold", y=1.01, color=DORANGE)
    # -- Churn pie
    counts = df[target_col].value_counts()
    axes[0].pie(counts, labels=counts.index, colors=[LBLUE, ORANGE],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2})
    axes[0].set_title("Churn Overall Distribution", fontweight="bold")
    # -- Churn distribution by categorical features
    for i, col in enumerate(cat_cols):
        print(col)
        ct = pd.crosstab(df[col], df[target_col], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=axes[i+1], 
                color=[ORANGE, LBLUE], edgecolor="white", legend=False)
        axes[i+1].set_title(f"Churn by {col}", fontweight="bold")
        axes[i+1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("outputs/03_eda_churn_by_cat.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✔ Saved: outputs/03_eda_churn_by_cat.png")

    # ─ Cat correlation:
    df["Churned"] = (df["attrition_flag"] == "Attrited Customer").astype(int)    
    BASELINE = df["Churned"].mean() * 100
    print(f"Baseline churn: {BASELINE:.1f}%\n")
    cat_results = {}
    for col in cat_cols:
        ct   = pd.crosstab(df[col], df["Churned"])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        rates = (df.groupby(col)["Churned"].mean() * 100).sort_values(ascending=False)
        cat_results[col] = {
            "chi2": chi2, "p": p, "cramers_v": cramers_v,
            "rates": rates, "sig": sig
        }
        print(f"\n  {col}  χ²={chi2:.1f}  V={cramers_v:.3f}  p{sig}")
        for cat, rate in rates.items():
            marker = " ▲" if rate > BASELINE else (" ▼" if rate < BASELINE else "  ")
            print(f"    {cat:<25} {rate:.1f}%{marker}")
        with open("outputs/cat_correlation.json", "w") as f:
            json.dump(cat_results, f, indent=2, default=str)
            print("Cat.correlation results saved to outputs/cat_correlation.json")
    
    # ─ Figure 1b:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[-1, -1].set_visible(False)   # hide 6th slot
    fig.suptitle("Churn Rate by Categorical Feature — with χ² Test Results",
             fontsize=13, fontweight="bold")
 
    income_order = ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Unknown"]
    edu_order    = ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate","Unknown"]
    orders       = {
        "Income_Category": income_order,
        "Education_Level": edu_order,
    }
 
    for ax, col in zip(axes.flatten(), cat_cols):
        res   = cat_results[col]
        rates = res["rates"]
        if col in orders:
            rates = rates.reindex([o for o in orders[col] if o in rates.index]).dropna()
 
        bar_colors = [RED if r > BASELINE else BLUE for r in rates.values]
        bars = ax.bar(rates.index, rates.values, color=bar_colors,
                  edgecolor="white", alpha=0.85, width=0.65)
        ax.axhline(BASELINE, color="#334155", linestyle="--", lw=1.2)
        ax.set_title(f"{col.replace('_',' ')}\nχ²={res['chi2']:.1f}  V={res['cramers_v']:.3f}  p{res['sig']}",
                 fontsize=10)
        ax.set_ylabel("Churn Rate (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xticklabels(rates.index, rotation=25, ha="right", fontsize=8)
        for bar, rate in zip(bars, rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{rate:.1f}%", ha="center", fontsize=8, fontweight="bold")
 
    plt.tight_layout()
    plt.savefig("outputs/04_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✔ Saved: outputs/04_categorical_churn_rates.png")

# ------------------------------------------------------------------
# STEP 6 — ANALYSIS 5: CHURN ANALYSIS BY NUMERICAL VARIABLES
# ------------------------------------------------------------------
def churn_by_num(
    df: pd.DataFrame, 
    target_col: str 
) -> None:
    print("=" * 65)
    print("STEP 6 : Churn by numerical variables")
    print("=" * 65)
    num_cols = df.select_dtypes(include='number').columns.tolist()
    num_cols.remove("Churned")
    df["Churned"] = (df["attrition_flag"] == "Attrited Customer").astype(int)    
    BASELINE = df["Churned"].mean() * 100
    print(f"Dataset: {len(df):,} customers  |  Baseline churn: {BASELINE:.1f}%\n")
    numeric_results = []
    for col in num_cols:
        g0 = df[df.Churned == 0][col].dropna()
        g1 = df[df.Churned == 1][col].dropna()
        _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        r, _ = stats.pointbiserialr(df["Churned"], df[col].fillna(df[col].median()))
        delta_pct = (g1.mean() - g0.mean()) / g0.mean() * 100
 
        numeric_results.append({
            "feature": col, "label": col,
            "mean_exist": g0.mean(), "mean_churn": g1.mean(),
            "med_exist":  g0.median(), "med_churn": g1.median(),
            "p_value": p, "r": r, "delta_pct": delta_pct,
        })
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        dir_arrow = "↑" if r > 0 else "↓"
        print(f"  {col:<32}  r={r:+.3f} {dir_arrow}  "
          f"Δmean={delta_pct:+.1f}%  p{sig}")
    res_df = pd.DataFrame(numeric_results).sort_values("r", key=abs, ascending=False)
    
    # - Figure 2a:
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [RED if r < 0 else GREEN for r in res_df["r"]]
    bars = ax.barh(res_df["label"], res_df["r"], color=colors,
               edgecolor="white", alpha=0.85, height=0.65)
    ax.axvline(0, color="#334155", lw=1)
    ax.set_xlabel("Point-Biserial Correlation r (with Churned)")
    ax.set_title("Bivariate Correlation — All Numeric Features vs Churn",
             fontsize=12, fontweight="bold", pad=12)
    for bar, val in zip(bars, res_df["r"]):
        xpos = val + (0.005 if val >= 0 else -0.005)
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f"{val:+.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8.5, fontweight="bold")
    red_p  = mpatches.Patch(color=RED,   label="Negative r  (higher → less churn)")
    grn_p  = mpatches.Patch(color=GREEN, label="Positive r  (higher → more churn)")
    ax.legend(handles=[red_p, grn_p], loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/05_eda_correlation_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✔ Saved: outputs/05_eda_correlation_waterfall.png")

    # - Figure 2b:
    TOP9 = res_df.head(9)["feature"].tolist()
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle("Distribution Comparison — Existing vs Attrited (Top 9 Features by |r|)",
             fontsize=14, fontweight="bold", color=DORANGE)
 
    for ax, col in zip(axes.flatten(), TOP9):
        data = [df[df.Churned==0][col].dropna(), df[df.Churned==1][col].dropna()]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5, notch=False,
                    orientation='horizontal',
                    medianprops={"color":"white","linewidth":2})
        for patch, c in zip(bp["boxes"], [LBLUE, ORANGE]):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        ax.set_yticklabels(["Existing", "Attrited"])
        ax.set_title(col, fontweight="bold")
        # annotate median diff
        m0, m1 = data[0].median(), data[1].median()
        pct = (m1 - m0) / m0 * 100 if m0 != 0 else 0
        ax.set_xlabel(f"Median Δ = {pct:+.1f}%", fontsize=10, labelpad=2)
 
    plt.tight_layout()
    plt.savefig("outputs/06_eda_boxplots_top9.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved: outputs/06_eda_boxplots_top9.png")

    # ─ Figure 2c:
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink":0.8}, ax=ax, annot_kws={"size":8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("outputs/07_eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved: outputs/07_eda_correlation_heatmap.png")

    # ─ Figure 2d:
    n_plots = len(num_cols)
    cols = 3
    rows = math.ceil(n_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("CREDIT CARD CHURN BY NUMERICAL COLUMNS", fontsize=16, fontweight="bold", y=1.01, color=DORANGE)
    for i, col in enumerate(num_cols):
        sns.histplot(data=df, x=col, hue=target_col, ax=axes[i], color=[LBLUE,ORANGE])
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig("outputs/08_eda_churn_by_num_hist.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✔ Saved: outputs/08_eda_churn_by_num_hist.png")



if __name__ == "__main__":
    # -------------------------------------
    # PIPELINE ORCHESTRATION
    # -------------------------------------
    # 01 - data loading
    df = load_data("data/cleaned_data.csv")
    # 02 - basic statistics
    cat_stats, num_stats = get_basic_statistics(df)
    with open("outputs/cat_stats.json", "w") as f:
        json.dump(cat_stats, f, indent=2, default=str)
    with open("outputs/num_stats.json", "w") as f:
        json.dump(num_stats, f, indent=2, default=str)
    # 03 - Univariate analysis: numerical distribution
    plot_numeric_distribution(df, cols=None)
    # 04 - Univariate analysis: categorical distribution
    plot_categorical_counts(df, cols=None)
    # 05 - churn by categorical variables
    churn_by_cat(df, "attrition_flag")
    # 06 - churn by numerical variables
    churn_by_num(df, "attrition_flag")

    print("\nChurn EDA completed!\n")
