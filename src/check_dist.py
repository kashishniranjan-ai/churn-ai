import sqlite3
import pandas as pd

def check_distribution():
    db_path = "data/processed/retail.db"
    conn = sqlite3.connect(db_path)

    with open("sql/rfm_analysis.sql", "r") as f:
        rfm_query = f.read()

    df_rfm = pd.read_sql_query(rfm_query, conn)
    
    print("R_Score Distribution:")
    print(df_rfm['R_Score'].value_counts().sort_index())
    
    print("\nF_Score Distribution:")
    print(df_rfm['F_Score'].value_counts().sort_index())
    
    print("\nM_Score Distribution:")
    print(df_rfm['M_Score'].value_counts().sort_index())
    
    print("\nMin/Max Frequency per F_Score:")
    print(df_rfm.groupby('F_Score')['Frequency'].agg(['min', 'max', 'count']))

if __name__ == "__main__":
    check_distribution()
