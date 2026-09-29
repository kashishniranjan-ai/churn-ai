import urllib.request
import pandas as pd
import os

print("Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
os.makedirs("data/raw", exist_ok=True)
excel_path = "data/raw/online_retail.xlsx"
csv_path = "data/raw/online_retail.csv"

if not os.path.exists(csv_path):
    urllib.request.urlretrieve(url, excel_path)
    print("Download complete. Converting to CSV...")
    df = pd.read_excel(excel_path)
    df.to_csv(csv_path, index=False, encoding="ISO-8859-1")
    print("Conversion complete.")
else:
    print("Dataset already exists.")
