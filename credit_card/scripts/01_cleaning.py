import numpy as np
import pandas as pd
from datetime import datetime
import warnings, os, json, re 

# ─ set up
warnings.filterwarnings("ignore")
os.makedirs("../outputs", exist_ok=True)
os.makedirs("../logs", exist_ok=True)

INPUT = "../data/credit_card_churn.csv"
OUTPUT = "../data/cleaned_data.csv"
LOG_PATH = "../logs/01_cleaning_log.json"

# - Sentinel strings treated as 
# missing in categorical columns
SENTINEL_VALUES = {
    "unknown", "n/a", "na", "nan", "none", "null", "",
    "missing", "not available", "not specified", "-", "?",
}
# - AUDIT LOG
log = {
    "run_at":       datetime.utcnow().isoformat() + "Z",
    "input_path":   INPUT,
    "output_path":  OUTPUT,
    "steps":        [],
}

def record(step, detail, rows_affected=0, cols_affected=None):
    entry = {"step": step, "detail": detail, "rows_affected": rows_affected}
    if cols_affected is not None:
        entry["cols_affected"] = cols_affected
    log["steps"].append(entry)
    tag = f"[{step}]"
    print(f"  {tag:<30} {detail}"
          + (f"  ({rows_affected} rows)" if rows_affected else "")
          + (f"  cols={cols_affected}"   if cols_affected  else ""))

# --------------------------------------------------------------------------------------
# STEP 1 — LOAD & SNAPSHOT
# --------------------------------------------------------------------------------------
print("=" * 65)
print("STEP 1 : Load and Snapshot")
print("=" * 65)
 
df = pd.read_csv(INPUT)
orig_shape = df.shape
record("load", 
    f"Read {orig_shape[0]:,} rows x {orig_shape[1]} cols from {INPUT}")
print(f"\nShape   : {df.shape}")
print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"  Dtypes  :\n{df.dtypes.to_string()}\n")
 
log["input_shape"]  = list(orig_shape)
log["input_dtypes"] = df.dtypes.astype(str).to_dict()

# --------------------------------------------------------------------------------------
# STEP 2 — DROP SPECIFIED COLUMNS
# --------------------------------------------------------------------------------------
print("=" * 65)
print("STEP 2 : Drop specified columns")
print("=" * 65)

DROP_COLS = ["CLIENTNUM", 
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
]

if DROP_COLS:
    df.drop(columns=DROP_COLS, inplace=True)
    record("force_drop", f"Dropped {len(DROP_COLS)} column(s)", cols_affected=DROP_COLS)
else:
    record("force_drop", "No columns configured for forced drop")


# --------------------------------------------------------------------------------------
# STEP 3 — COLUMN NAME NORMALIZATION
# --------------------------------------------------------------------------------------
print("=" * 65)
print("STEP 3 : Column name normalisation")
print("=" * 65)
 
def to_snake(name):
    name = name.strip()
    #spaces/hyphens -> underscore
    name = re.sub(r"[\s\-]+", "_", name)
    # drop non-word chars 
    name = re.sub(r"[^\w]", "", name)
    # camelCase → snake
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name) 
    return name.lower()

renamed = {c: to_snake(c) for c in df.columns if to_snake(c) != c}
if renamed:
    df.rename(columns=renamed, inplace=True)
    record("col_normalization", f"Normalized {len(renamed)} columns", cols_affected=list(renamed.keys()))
else:
    record("col_normalization", "All column names already clean — no changes")

# --------------------------------------------------------------------------------------
# STEP 4 — DUPLICATES
# --------------------------------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 4 : Duplicate handling")
print("=" * 65)
 
n_dupes = df.duplicated().sum()
if n_dupes:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    record("drop_duplicates", f"Removed {n_dupes} fully duplicate rows", rows_affected=int(n_dupes))
else:
    record("drop_duplicates", "No duplicate rows found")

# --------------------------------------------------------------------------------------
# STEP 5 — CONSTANT COLUMNS
# --------------------------------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 5 : Constant columns")
print("=" * 65)

const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
if const_cols:
    df.drop(columns=const_cols, inplace=True)
    record("drop_constant_cols", f"Dropped {len(const_cols)} constant column(s)", cols_affected=const_cols)
else:
    record("drop_constant_cols", "No constant columns found")

# --------------------------------------------------------------------------------------
# STEP 6 — MISSING VALUES
# --------------------------------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 6 : Missing values")
print("=" * 65)

null_counts = df.isnull().sum()
null_cols   = null_counts[null_counts > 0]
 
if null_cols.empty:
    record("null_values", "No true nulls found in any column")
else:
    record("null_values", f"{len(null_cols)} column(s) contain NaN",
        cols_affected=null_counts[null_counts > 0].to_dict())
    str_null_cols = [c for c in null_cols.index
                     if df[c].dtype.kind in ("O", "U", "S")
                     or pd.api.types.is_string_dtype(df[c])]
    num_null_cols = [c for c in null_cols.index if c not in str_null_cols]
    print("String null cols:", str_null_cols)
    print("Numeric null cols:", num_null_cols)

# --------------------------------------------------------------------------------------
# STEP 7 — STRING NORMALIZATION
# --------------------------------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 7 : String normalisation")
print("=" * 65)

TITLE_CASE_STRINGS = False
str_cols_final = df.select_dtypes(include=["object", "str"]).columns.tolist()
for col in str_cols_final:
    df[col] = df[col].astype(str).str.strip()
    if TITLE_CASE_STRINGS:
        df[col] = df[col].str.title()
 
record("string_normalise",
       f"Stripped whitespace on {len(str_cols_final)} string column(s)"
       + (" + title-cased" if TITLE_CASE_STRINGS else ""),
       cols_affected=str_cols_final)

# --------------------------------------------------------------------------------------
# STEP 8 — OUTLIER DETECTION (IQR)
# --------------------------------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 8 : Outlier detection (IQR)")
print("=" * 65)

num_cols_final = df.select_dtypes(include=np.number).columns.tolist()
outlier_report = {}
 
for col in num_cols_final:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    if IQR == 0:
        continue                         # binary / near-constant — skip
    lo  = Q1 - 1.5 * IQR
    hi  = Q3 + 1.5 * IQR
    mask = (df[col] < lo) | (df[col] > hi)
    n   = int(mask.sum())
    if n:
        outlier_report[col] = {
            "n_outliers": n,
            "pct": round(n / len(df) * 100, 2),
            "lower_fence": round(float(lo), 4),
            "upper_fence": round(float(hi), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
        }
 
if outlier_report:
    for col, info in outlier_report.items():
        record("outlier_flag",
               f"{col}: {info['n_outliers']} outliers ({info['pct']}%) "
               f"outside [{info['lower_fence']}, {info['upper_fence']}]")
    print("\n NOTE: Outliers are flagged only. "
          "Set FORCE_DROP_COLS or add a capping step if removal is needed.")
else:
    record("outlier_flag", "No IQR outliers detected in any numeric column")
 
log["outlier_report"] = outlier_report

# --------------------------------------------------------------------------------------
# STEP 9 — FINAL VALIDATION & SAVE
# --------------------------------------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 9 : Final validation & save")
print("=" * 65)

final_shape     = df.shape
log["rows_removed"]          = orig_shape[0] - final_shape[0]
log["cols_removed"]          = orig_shape[1] - final_shape[1]
log["output_dtypes"]         = df.dtypes.astype(str).to_dict()

df.to_csv(OUTPUT, index=False)
 
with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2, default=str)

# --------------------------------------------------------------------------------------
# SUMMARY REPORT
# --------------------------------------------------------------------------------------

print(f"""
  ┌─────────────────────────────────────────────┐
  │   Cleaning complete                         │
  ├─────────────────────────────────────────────┤
  │   Input shape   : {str(orig_shape):<28}     │
  │   Output shape  : {str(final_shape):<28}    │
  │   Rows removed  : {log['rows_removed']:<28} │
  │   Cols removed  : {log['cols_removed']:<28} │
  │   Steps run     : {len(log['steps']):<28}   │
  ├─────────────────────────────────────────────┤
  │   Saved → {OUTPUT:<35}                      │
  │   Log   → {LOG_PATH:<35}                    │
  └─────────────────────────────────────────────┘
""")

