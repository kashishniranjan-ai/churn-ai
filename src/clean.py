import pandas as pd
import os

def clean_data(input_path: str, clean_out_path: str, returns_out_path: str):
    """
    Cleans the UCI Online Retail dataset.
    """
    print(f"Loading data from {input_path}...")
    try:
        # Rule 5: Read the CSV with encoding="ISO-8859-1"
        df = pd.read_csv(input_path, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")

    # 1. Remove missing Customer IDs
    df_cleaned = df.dropna(subset=['CustomerID']).copy()
    rows_dropped = initial_rows - len(df_cleaned)
    print(f"Removed {rows_dropped} rows due to missing CustomerID.")

    # Rule 3: Cast CustomerID to int
    df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(int)

    # Rule 3: Strip Description
    df_cleaned['Description'] = df_cleaned['Description'].str.strip()

    # Rule 3: Parse InvoiceDate to datetime
    df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

    # Rule 4: Extract cancellations and save to returns.csv
    df_cleaned['InvoiceNo'] = df_cleaned['InvoiceNo'].astype(str)
    cancellations = df_cleaned[df_cleaned['InvoiceNo'].str.startswith('C')]
    
    # Save returns separately
    os.makedirs(os.path.dirname(returns_out_path), exist_ok=True)
    cancellations.to_csv(returns_out_path, index=False)
    print(f"Saved {len(cancellations)} cancellation rows to {returns_out_path}")

    # Remove cancellations from clean data
    curr_rows = len(df_cleaned)
    df_cleaned = df_cleaned[~df_cleaned['InvoiceNo'].str.startswith('C')]
    print(f"Removed {curr_rows - len(df_cleaned)} rows due to cancellations.")

    # Rule 1: Drop invalid rows (quantity <= 0, price <= 0)
    curr_rows = len(df_cleaned)
    df_cleaned = df_cleaned[(df_cleaned['Quantity'] > 0) & (df_cleaned['UnitPrice'] > 0)]
    rows_dropped = curr_rows - len(df_cleaned)
    print(f"Removed {rows_dropped} rows due to invalid quantity or price (<= 0).")

    # Remove duplicates
    curr_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    rows_dropped = curr_rows - len(df_cleaned)
    print(f"Removed {rows_dropped} duplicate rows.")

    # Rule 2: Exclude non-product StockCodes
    curr_rows = len(df_cleaned)
    # Using regex to find codes that are strictly alphabets and spaces 
    # (e.g. POST, DOT, M, BANK CHARGES, AMAZONFEE)
    non_product_mask = df_cleaned['StockCode'].astype(str).str.contains(r'^[a-zA-Z\s]+$', regex=True, na=False)
    df_cleaned = df_cleaned[~non_product_mask]
    rows_dropped = curr_rows - len(df_cleaned)
    print(f"Removed {rows_dropped} rows with non-product StockCodes.")

    # Rule 3: Add TotalPrice
    df_cleaned['TotalPrice'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

    # Rule 6: Final Summary
    final_rows = len(df_cleaned)
    unique_customers = df_cleaned['CustomerID'].nunique()
    date_min = df_cleaned['InvoiceDate'].min()
    date_max = df_cleaned['InvoiceDate'].max()

    print("\n=== FINAL SUMMARY ===")
    print(f"Rows before: {initial_rows}")
    print(f"Rows after: {final_rows}")
    print(f"Unique Customers: {unique_customers}")
    print(f"Date Range: {date_min} to {date_max}")

    # Save cleaned dataset
    os.makedirs(os.path.dirname(clean_out_path), exist_ok=True)
    df_cleaned.to_csv(clean_out_path, index=False)
    print(f"Cleaned data saved to {clean_out_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/raw/online_retail.csv"
    CLEAN_DATA_PATH = "data/processed/clean_retail.csv"
    RETURNS_DATA_PATH = "data/processed/returns.csv"
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}.")
    else:
        clean_data(RAW_DATA_PATH, CLEAN_DATA_PATH, RETURNS_DATA_PATH)
