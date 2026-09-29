import sqlite3
import pandas as pd
import os

def create_database():
    csv_path = "data/processed/clean_retail.csv"
    db_path = "data/processed/retail.db"
    
    print(f"Loading {csv_path} into {db_path}...")
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    
    # Cast InvoiceDate back to datetime for SQLite if needed, but string is fine
    
    conn = sqlite3.connect(db_path)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    
    # Create indexes to speed up the analytical queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customer ON transactions(CustomerID)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON transactions(InvoiceDate)")
    
    print("Database created successfully with 'transactions' table.")
    conn.close()

if __name__ == "__main__":
    create_database()
