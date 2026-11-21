"""
Phase 2: Preprocessing and Feature Engineering
Dataset: Forbes Most Valuable Sports Teams
Author: ASBD Hackathon Team
Random State: 42
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def parse_price(val_str):
    """
    Parses various currency strings into numeric float in millions USD ($M).
    Handles expressions like '$150 million', '$1.4 billion', '$20 million, $268 million',
    '$180,000', '$100', 'not applicable'.
    """
    if pd.isna(val_str):
        return np.nan
    val_str = str(val_str).strip().lower()
    if val_str in ['not applicable', 'n/a', 'na', 'none', '']:
        return np.nan

    # If multiple values exist (e.g. "$20 million, $268 million"), sum them
    parts = val_str.split(',')
    # Check if comma was thousands separator or multi-purchase delimiter
    # If the text has words like 'million' or 'billion' in separate parts
    if any('million' in p or 'billion' in p for p in parts) and len(parts) > 1 and any(re.search(r'\d+', p) for p in parts):
        # Could be "$500, $150 million" or "$20 million, $268 million"
        total = 0.0
        # Check if the parts are separate amounts
        for p in parts:
            p = p.strip()
            total += parse_single_price(p)
        return total
    else:
        return parse_single_price(val_str)

def parse_single_price(s):
    """Parse single amount string to USD Millions"""
    s = s.replace('$', '').replace(',', '').strip()
    if 'billion' in s:
        num = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
        return float(num[0]) * 1000.0 if num else np.nan
    elif 'million' in s:
        num = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
        return float(num[0]) if num else np.nan
    else:
        # Plain number in dollars (e.g. 100, 180000, 2500)
        num = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
        if num:
            return float(num[0]) / 1_000_000.0 # Convert dollars to millions
        return np.nan

def parse_year(val_str):
    """
    Parses year string into integer year.
    If multiple years (e.g. '1979, 1998'), returns initial acquisition year.
    """
    if pd.isna(val_str):
        return np.nan
    val_str = str(val_str).strip().lower()
    if val_str in ['not applicable', 'n/a', 'na', 'none', '']:
        return np.nan
    matches = re.findall(r"\b(19\d\d|20\d\d)\b", val_str)
    if matches:
        return float(matches[0]) # Initial year
    return np.nan

def parse_value_b(val_str):
    """Extracts numeric valuation in Billions USD"""
    if pd.isna(val_str):
        return np.nan
    s = str(val_str).replace('$', '').replace('billion', '').replace(',', '').strip()
    return float(s)

def parse_change_pct(val_str):
    """Extracts numeric 5-year percentage change"""
    if pd.isna(val_str):
        return np.nan
    s = str(val_str).replace('%', '').strip()
    return float(s)

def decompose_team_sport(team_str):
    """
    Extracts team name, sport, league, and country from string like 'Dallas Cowboys (NFL)'
    """
    team_str = str(team_str).strip()
    match = re.search(r"^(.*?)\s*\((.*?)\)$", team_str)
    if match:
        name = match.group(1).strip()
        league_sport = match.group(2).strip()
    else:
        name = team_str
        league_sport = "Unknown"

    # Determine Sport, League, Country
    if league_sport == "NFL":
        sport = "American Football"
        league = "NFL"
        country = "USA"
    elif league_sport == "MLB":
        sport = "Baseball"
        league = "MLB"
        country = "USA"
    elif league_sport == "NBA":
        sport = "Basketball"
        league = "NBA"
        country = "USA"
    elif league_sport == "Soccer":
        sport = "Soccer"
        # European soccer leagues & countries mapping
        soccer_mapping = {
            "Barcelona": ("La Liga", "Spain"),
            "Real Madrid": ("La Liga", "Spain"),
            "Bayern Munich": ("Bundesliga", "Germany"),
            "Manchester United": ("Premier League", "UK"),
            "Liverpool": ("Premier League", "UK"),
            "Manchester City": ("Premier League", "UK"),
            "Chelsea": ("Premier League", "UK"),
            "Arsenal": ("Premier League", "UK"),
            "Paris Saint-Germain": ("Ligue 1", "France"),
        }
        league, country = soccer_mapping.get(name, ("European Soccer", "Europe"))
    else:
        sport = league_sport
        league = league_sport
        country = "USA"

    return name, sport, league, country

def main():
    print("=" * 80)
    print("PHASE 2: PREPROCESSING, SELECTION & TRANSFORMATION")
    print("=" * 80)

    raw_path = os.path.join("data", "raw_sports_teams.csv")
    df = pd.read_csv(raw_path)

    # 1. Deduplication
    print("\n--- 1. CLEANING & DEDUPLICATION ---")
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates dropped: {initial_rows - len(df)} (Retained {len(df)} rows)")

    # 2. Decompose Team string
    print("\n--- 2. CATEGORY EXTRACTION & STANDARDIZATION ---")
    parsed_info = df['Sports Team'].apply(decompose_team_sport)
    df['Team_Name'] = [p[0] for p in parsed_info]
    df['Sport'] = [p[1] for p in parsed_info]
    df['League'] = [p[2] for p in parsed_info]
    df['Country'] = [p[3] for p in parsed_info]

    print("Sample extracted categories:")
    print(df[['Sports Team', 'Team_Name', 'Sport', 'League', 'Country']].head(6))

    # 3. Numeric string conversions
    print("\n--- 3. NUMERIC EXTRACTION (Stripping $, B, M, %) ---")
    df['Value_USD_B'] = df['Value'].apply(parse_value_b)
    df['Five_Year_Change_Pct'] = df['Five-Year Change In Value'].apply(parse_change_pct)
    df['Year_Purchased_Clean'] = df['Year Purchased'].apply(parse_year)
    df['Price_Paid_USD_M'] = df['Price Paid'].apply(parse_price)

    # 4. Missing value imputation
    print("\n--- 4. MISSING VALUE HANDLING (Median / Mode) ---")
    print("Nulls before imputation:")
    print(df[['Value_USD_B', 'Five_Year_Change_Pct', 'Year_Purchased_Clean', 'Price_Paid_USD_M']].isnull().sum())

    # For fan-owned soccer clubs (Barcelona, Real Madrid, Bayern Munich), acquisition year & price are 'not applicable'
    # Impute with median
    median_year = df['Year_Purchased_Clean'].median()
    median_price = df['Price_Paid_USD_M'].median()
    print(f"Median Year Purchased: {median_year:.0f} | Median Price Paid ($M): {median_price:.2f}")

    # Track imputation flag
    df['Price_Imputed'] = df['Price_Paid_USD_M'].isna().astype(int)
    df['Year_Imputed'] = df['Year_Purchased_Clean'].isna().astype(int)

    df['Year_Purchased_Clean'] = df['Year_Purchased_Clean'].fillna(median_year)
    df['Price_Paid_USD_M'] = df['Price_Paid_USD_M'].fillna(median_price)
    
    # Calculate derived financial indicators
    CURRENT_YEAR = 2021 # Forbes list vintage
    df['Years_Held'] = CURRENT_YEAR - df['Year_Purchased_Clean']
    df['Price_Paid_USD_B'] = df['Price_Paid_USD_M'] / 1000.0
    # Annualized growth or value multiple
    df['Valuation_Gain_USD_B'] = df['Value_USD_B'] - df['Price_Paid_USD_B']

    # 5. Outlier Detection with IQR (Flagging, not blindly dropping)
    print("\n--- 5. OUTLIER DETECTION (IQR Flagging) ---")
    numeric_cols = ['Value_USD_B', 'Five_Year_Change_Pct', 'Year_Purchased_Clean', 'Price_Paid_USD_M', 'Years_Held']
    outlier_summary = {}

    for col in numeric_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        flag_col = f'{col}_Outlier'
        df[flag_col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
        outlier_count = df[flag_col].sum()
        outlier_summary[col] = {
            'Q1': q25, 'Q3': q75, 'IQR': iqr, 
            'Lower': lower_bound, 'Upper': upper_bound, 
            'Outlier_Count': outlier_count
        }
        print(f"Column: {col:<22} | Bounds: [{lower_bound:8.2f}, {upper_bound:8.2f}] | Outliers: {outlier_count}")

    # 6. Feature Selection
    print("\n--- 6. FEATURE SELECTION & VARIANCE CHECK ---")
    # Drop raw unparsed and redundant ID columns
    drop_cols = ['Unnamed: 0', 'Sports Team', 'Value', 'Five-Year Change In Value', 'Year Purchased', 'Price Paid']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Check variance of numerical features
    print("Numerical Variance:")
    print(df_clean[numeric_cols].var())

    # Check correlation matrix
    print("\nPearson Correlation Matrix:")
    corr_matrix = df_clean[numeric_cols].corr()
    print(corr_matrix.round(3))

    # Save clean dataset
    clean_path = os.path.join("data", "clean.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"\n[OK] Cleaned dataset saved to: {clean_path} (Shape: {df_clean.shape})")

    # 7. Discretization & Transformation for Data Mining
    print("\n--- 7. DISCRETIZATION (Low/Med/High) & TRANSACTION GENERATION ---")
    # Bin continuous variables into 3 quantiles (Low, Med, High)
    df_trans = pd.DataFrame()
    df_trans['Sport'] = 'Sport_' + df_clean['Sport'].astype(str)
    df_trans['League'] = 'League_' + df_clean['League'].astype(str)
    df_trans['Country'] = 'Country_' + df_clean['Country'].astype(str)
    
    # Value tiers
    df_trans['Value_Tier'] = pd.qcut(df_clean['Value_USD_B'], q=3, labels=['Value_Low', 'Value_Med', 'Value_High'])
    # 5-Year Change tiers
    df_trans['Growth_Tier'] = pd.qcut(df_clean['Five_Year_Change_Pct'], q=3, labels=['Growth_Low', 'Growth_Med', 'Growth_High'])
    # Price Paid tiers
    df_trans['Price_Tier'] = pd.qcut(df_clean['Price_Paid_USD_M'], q=3, labels=['PricePaid_Low', 'PricePaid_Med', 'PricePaid_High'])
    # Tenure tiers
    df_trans['Tenure_Tier'] = pd.qcut(df_clean['Years_Held'], q=3, labels=['Tenure_Short', 'Tenure_Med', 'Tenure_Long'])

    # Build transaction table for FP-Growth / CHARM
    # Each row is a transaction list of categorical tokens
    transactions = df_trans.values.tolist()
    
    # Save transactions.csv in format: TID, Items (comma-separated)
    trans_df = pd.DataFrame({
        'TID': range(len(transactions)),
        'Items': [", ".join(t) for t in transactions]
    })
    trans_path = os.path.join("data", "transactions.csv")
    trans_df.to_csv(trans_path, index=False)
    print(f"[OK] Transactions table saved to: {trans_path} (50 transactions)")

    # 8. Feature Scaling (Min-Max and Z-score)
    print("\n--- 8. FEATURE SCALING (Min-Max & Standard Scaler) ---")
    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()

    scaled_std = scaler_std.fit_transform(df_clean[numeric_cols])
    scaled_mm = scaler_minmax.fit_transform(df_clean[numeric_cols])

    df_scaled = df_clean.copy()
    for i, col in enumerate(numeric_cols):
        df_scaled[f'{col}_zscore'] = scaled_std[:, i]
        df_scaled[f'{col}_minmax'] = scaled_mm[:, i]

    scaled_path = os.path.join("data", "scaled.csv")
    df_scaled.to_csv(scaled_path, index=False)
    print(f"[OK] Scaled dataset saved to: {scaled_path} (Shape: {df_scaled.shape})")

    print("\n[PHASE 2 COMPLETE] Data cleaning, feature selection, scaling and transactions generated.")

if __name__ == "__main__":
    main()
