import sqlite3
import pandas as pd

def validate_sql():
    db_path = "data/processed/retail.db"
    conn = sqlite3.connect(db_path)

    # Load RFM Query
    with open("sql/rfm_analysis.sql", "r") as f:
        rfm_query = f.read()

    # Load Cohort Query
    with open("sql/cohort_retention.sql", "r") as f:
        cohort_query = f.read()

    # Execute RFM
    print("--- RFM Output (First 10 Rows) ---")
    df_rfm = pd.read_sql_query(rfm_query, conn)
    print(df_rfm.head(10).to_string())

    # Execute Cohort
    print("\n--- Cohort Retention Output (First 10 Rows) ---")
    df_cohort = pd.read_sql_query(cohort_query, conn)
    print(df_cohort.head(10).to_string())

    print("\n=== VALIDATION CHECKS ===")

    # 1. Check 4334 rows in RFM
    rfm_rows = len(df_rfm)
    print(f"RFM Rows: {rfm_rows} (Expected 4334) -> {'PASS' if rfm_rows == 4334 else 'FAIL'}")

    # 2. Check Month 0 retention = 100%
    month0 = df_cohort[df_cohort['MonthIndex'] == 0]
    all_100 = (month0['RetentionRate'] == 100.0).all()
    print(f"Month 0 Retention == 100%: -> {'PASS' if all_100 else 'FAIL'}")

    # 3. Check Monetary total matches clean data
    rfm_monetary = df_rfm['Monetary'].sum()
    df_clean = pd.read_sql_query("SELECT SUM(TotalPrice) as tot FROM transactions", conn)
    clean_monetary = df_clean.iloc[0]['tot']
    
    # We use round to avoid small floating point differences
    print(f"RFM Total Monetary: {rfm_monetary:.2f}")
    print(f"Clean Total Monetary: {clean_monetary:.2f}")
    
    if round(rfm_monetary, 2) == round(clean_monetary, 2):
        print("Monetary Totals Match -> PASS")
    else:
        print("Monetary Totals Match -> FAIL")
        
    # 4. Check every score value 1-5 appears
    for col in ['R_Score', 'F_Score', 'M_Score']:
        unique_scores = sorted(df_rfm[col].unique())
        print(f"{col} unique scores: {unique_scores} -> {'PASS' if unique_scores == [1, 2, 3, 4, 5] else 'FAIL'}")

    # 5. Check Frequency ranges don't overlap across F_Scores
    f_ranges = df_rfm.groupby('F_Score')['Frequency'].agg(['min', 'max'])
    overlap = False
    prev_max = -1
    for score in sorted(f_ranges.index):
        curr_min = f_ranges.loc[score, 'min']
        if curr_min <= prev_max:
            overlap = True
        prev_max = f_ranges.loc[score, 'max']
    print(f"Frequency Ranges Overlap: -> {'FAIL' if overlap else 'PASS (No overlap)'}")

    conn.close()

if __name__ == "__main__":
    validate_sql()
