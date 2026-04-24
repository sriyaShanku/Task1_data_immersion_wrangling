import os
import pandas as pd
import numpy as np

# ----------------------------
# PATHS
# ----------------------------
INPUT_FILE = "data/raw/dataset.csv"
OUTPUT_FILE = "data/processed/cleaned_dataset.csv"

OUTPUT_DIR = "outputs"
PROFILE_FILE = os.path.join(OUTPUT_DIR, "data_profile.txt")
MISSING_FILE = os.path.join(OUTPUT_DIR, "missing_values.csv")
DUPLICATES_FILE = os.path.join(OUTPUT_DIR, "duplicates.csv")
OUTLIERS_FILE = os.path.join(OUTPUT_DIR, "outliers.csv")
DATA_DICT_FILE = os.path.join(OUTPUT_DIR, "data_dictionary.csv")

os.makedirs("data/processed", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(INPUT_FILE)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Columns:", df.columns)

# ----------------------------
# DATA PROFILE
# ----------------------------
profile = []
profile.append("DATASET PROFILE\n")
profile.append(f"Rows: {df.shape[0]}")
profile.append(f"Columns: {df.shape[1]}\n")

profile.append("COLUMN INFO:")
for col in df.columns:
    profile.append(f"- {col}: {df[col].dtype}")

profile.append("\nMISSING VALUES:")
missing = df.isnull().sum().sort_values(ascending=False)
profile.append(str(missing))

profile.append("\nDUPLICATES:")
profile.append(str(df.duplicated().sum()))

with open(PROFILE_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(profile))

# ----------------------------
# SAVE MISSING VALUES
# ----------------------------
missing_df = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isnull().sum().values,
    "missing_percent": (df.isnull().sum().values / len(df)) * 100
})
missing_df.to_csv(MISSING_FILE, index=False)

# ----------------------------
# REMOVE DUPLICATES
# ----------------------------
duplicate_rows = df[df.duplicated()]
duplicate_rows.to_csv(DUPLICATES_FILE, index=False)
df = df.drop_duplicates()

# ----------------------------
# CLEAN TEXT
# ----------------------------
for col in df.select_dtypes(include="object"):
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace("nan", np.nan)

# ----------------------------
# CONVERT DATES (SAFE)
# ----------------------------
if "order_date" in df.columns:
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

if "ship_date" in df.columns:
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")

# ----------------------------
# HANDLE MISSING VALUES
# ----------------------------
for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col] = df[col].fillna(df[col].median())
    elif "datetime" in str(df[col].dtype):
        df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
    else:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
# Year & Month
if "order_date" in df.columns:
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month_name()

# Profit margin
if "profit" in df.columns and "sales" in df.columns:
    df["profit_margin"] = df["profit"] / df["sales"]

# ----------------------------
# OUTLIER DETECTION + HANDLING
# ----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_rows = pd.DataFrame()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    if not outliers.empty:
        temp = outliers.copy()
        temp["outlier_column"] = col
        outlier_rows = pd.concat([outlier_rows, temp], ignore_index=True)

    # Cap values
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

outlier_rows.to_csv(OUTLIERS_FILE, index=False)

# ----------------------------
# FINAL CLEANUP
# ----------------------------
df.replace([np.inf, -np.inf], np.nan, inplace=True)

for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("Unknown")

# ----------------------------
# DATA DICTIONARY
# ----------------------------
data_dict = pd.DataFrame({
    "column_name": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "description": "",
    "example_value": [
        df[c].dropna().iloc[0] if df[c].dropna().shape[0] > 0 else ""
        for c in df.columns
    ],
    "business_relevance": ""
})
data_dict.to_csv(DATA_DICT_FILE, index=False)

# ----------------------------
# SAVE CLEAN DATA
# ----------------------------
df.to_csv(OUTPUT_FILE, index=False)

print("\nTask 1 Completed Successfully")
print(f"Cleaned dataset saved at: {OUTPUT_FILE}")
print(f"Outputs saved in: {OUTPUT_DIR}")