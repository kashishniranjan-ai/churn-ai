import pandas as pd

print("1. Unique StockCodes removed by the non-product filter with row counts:")
df = pd.read_csv("data/raw/online_retail.csv", encoding="ISO-8859-1")
df = df.dropna(subset=['CustomerID']).copy()
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df = df[~df['InvoiceNo'].str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df = df.drop_duplicates()

# Apply filter to find what was removed
mask = df['StockCode'].astype(str).str.contains(r'^[a-zA-Z\s]+$', regex=True, na=False)
removed = df[mask]
print(removed['StockCode'].value_counts())
print("\n")

print("2. Confirm clean_retail.csv structure:")
df_clean = pd.read_csv("data/processed/clean_retail.csv", encoding="ISO-8859-1")
print(df_clean[['CustomerID', 'InvoiceDate', 'TotalPrice']].head())
print("\nTypes:")
print(df_clean[['CustomerID', 'InvoiceDate', 'TotalPrice']].dtypes)
