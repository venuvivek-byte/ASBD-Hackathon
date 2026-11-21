"""
Phase 1: Load and Inspect Dataset
Dataset: Forbes Most Valuable Sports Teams
Author: ASBD Hackathon Team
Random State: 42
"""

import os
import shutil
import pandas as pd
import numpy as np

# Ensure deterministic execution
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    print("=" * 80)
    print("PHASE 1: DATA INGESTION AND EXPLORATORY INSPECTION")
    print("=" * 80)

    # 1. Standardize raw file location
    raw_source = os.path.join("data", "Most Values Sports Team")
    target_csv = os.path.join("data", "raw_sports_teams.csv")

    if os.path.exists(raw_source) and not os.path.exists(target_csv):
        shutil.copy(raw_source, target_csv)
        print(f"[OK] Copied raw data to standard CSV path: {target_csv}")
    elif os.path.exists(target_csv):
        print(f"[OK] Target raw CSV already exists: {target_csv}")
    else:
        raise FileNotFoundError(f"Source file not found at {raw_source}")

    # 2. Load dataset into pandas DataFrame
    # Note: Column 0 is an unnamed index in the raw file
    df = pd.read_csv(target_csv)
    print(f"\n[INFO] Dataset successfully loaded. Shape: {df.shape[0]} rows x {df.shape[1]} columns.\n")

    # 3. Basic DataFrame Info
    print("-" * 40)
    print("DATA SUMMARY: df.info()")
    print("-" * 40)
    df.info()

    # 4. Preview First 5 and Last 5 Rows
    print("\n" + "-" * 40)
    print("HEAD (Top 5 Rows):")
    print("-" * 40)
    print(df.head())

    print("\n" + "-" * 40)
    print("TAIL (Bottom 5 Rows):")
    print("-" * 40)
    print(df.tail())

    # 5. Column Names and Real Data Types
    print("\n" + "-" * 40)
    print("COLUMN NAMES AND REAL DTYPES:")
    print("-" * 40)
    for i, col in enumerate(df.columns):
        print(f"Col {i}: '{col}' | Type: {df[col].dtype} | Non-Null: {df[col].count()}/{len(df)}")

    # 6. Null Values and Missing Data Count
    print("\n" + "-" * 40)
    print("MISSING / NULL VALUE COUNTS:")
    print("-" * 40)
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().sum() / len(df)) * 100
    missing_summary = pd.DataFrame({'Missing Count': null_counts, 'Percentage (%)': null_pct})
    print(missing_summary)

    # Check for string-based missing representations like 'not applicable'
    print("\n" + "-" * 40)
    print("POTENTIAL MASKED MISSING VALUES ('not applicable', 'N/A', etc.):")
    print("-" * 40)
    for col in df.columns:
        masked_count = df[col].astype(str).str.strip().str.lower().isin(['not applicable', 'na', 'n/a', 'nan', 'none', '']).sum()
        if masked_count > 0:
            print(f"Column '{col}' has {masked_count} values matching 'not applicable' / missing string placeholders.")

    # 7. Summary Statistics for Object/Categorical Fields
    print("\n" + "-" * 40)
    print("DESCRIPTIVE SUMMARY: df.describe(include='all')")
    print("-" * 40)
    print(df.describe(include='all'))

    # 8. Real Columns Analysis
    print("\n" + "-" * 40)
    print("REAL ATTRIBUTES VERIFICATION & DOMAIN MAPPING:")
    print("-" * 40)
    print("Observed raw columns in the Forbes dataset:")
    print("1. 'Unnamed: 0': Integer row index (to be dropped in selection phase)")
    print("2. 'Sports Team': Team name with League/Sport appended in parentheses (e.g. 'Dallas Cowboys (NFL)', 'Barcelona (Soccer)')")
    print("   -> Action: Decompose into clean 'Team', 'League', 'Sport', and inferred 'Country'")
    print("3. 'Value': Valuation string (e.g. '$5.7 billion', '2.85 billion')")
    print("   -> Action: Strip '$' and 'billion', parse to numeric float (in billions USD)")
    print("4. 'Five-Year Change In Value': Percentage increase string (e.g. '43%')")
    print("   -> Action: Strip '%', parse to numeric float")
    print("5. 'Owner': Majority franchise owner(s) or entity (e.g. 'Jerry Jones', 'club members')")
    print("6. 'Year Purchased': Acquisition year (e.g. '1989', '1979, 1998', 'not applicable')")
    print("   -> Action: Clean multi-year entries, handle 'not applicable', convert to numeric")
    print("7. 'Price Paid': Acquisition cost string (e.g. '$150 million', '$1.4 billion', '$500, $150 million', '$100')")
    print("   -> Action: Convert all entries to uniform USD billions or millions scale, handle 'not applicable'")

    print("\n[PHASE 1 COMPLETE] Data successfully inspected and verified.")

if __name__ == "__main__":
    main()
