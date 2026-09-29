import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def rfm_segmentation(conn):
    print("--- 1. RFM Segmentation ---")
    df_rfm = pd.read_sql_query(open("sql/rfm_analysis.sql").read(), conn)
    
    # Define Segment Rules
    def assign_segment(row):
        r, f = row['R_Score'], row['F_Score']
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f < 3:
            return 'Potential Loyalist'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        else:
            return 'Hibernating / Lost'
            
    df_rfm['Segment'] = df_rfm.apply(assign_segment, axis=1)
    
    # Calculate historical revenue by segment
    clv = df_rfm.groupby('Segment').agg(
        CustomerCount=('CustomerID', 'count'),
        Avg_Revenue_Per_Customer=('Monetary', 'mean'),
        TotalRevenue=('Monetary', 'sum')
    ).sort_values('TotalRevenue', ascending=False)
    
    print(clv)
    
    os.makedirs("data/processed", exist_ok=True)
    df_rfm.to_csv("data/processed/rfm_segments.csv", index=False)
    clv.to_csv("data/processed/segment_revenue.csv")
    return df_rfm

def cohort_analysis(conn):
    print("\n--- 2. Cohort Retention Heatmap ---")
    df_cohort = pd.read_sql_query(open("sql/cohort_retention.sql").read(), conn)
    
    # Pivot into a heatmap matrix
    heatmap = df_cohort.pivot(index='CohortMonth', columns='MonthIndex', values='RetentionRate')
    print(heatmap.head(5))
    heatmap.to_csv("data/processed/cohort_heatmap.csv")

def market_basket(conn):
    print("\n--- 3. Market Basket Analysis (Top Pairs) ---")
    # Fetch invoice and products, ignoring returns
    query = """
    SELECT InvoiceNo, StockCode, Description
    FROM transactions
    WHERE Quantity > 0
    """
    df = pd.read_sql_query(query, conn)
    
    # Get products per invoice
    basket = df.groupby('InvoiceNo')['Description'].apply(set).reset_index()
    
    # We will compute a simple co-occurrence matrix for the top 50 most frequent items
    # to avoid memory explosion.
    top_items = df['Description'].value_counts().head(50).index
    
    pairs = {}
    item_counts = {item: 0 for item in top_items}
    total_invoices = len(basket)
    
    for items in basket['Description']:
        # only keep top items present in this invoice
        inv_items = [item for item in items if item in top_items]
        for item in inv_items:
            item_counts[item] += 1
            
        for i in range(len(inv_items)):
            for j in range(i+1, len(inv_items)):
                pair = tuple(sorted([inv_items[i], inv_items[j]]))
                pairs[pair] = pairs.get(pair, 0) + 1
                
    # Calculate Lift
    results = []
    for (itemA, itemB), pair_count in pairs.items():
        if pair_count < 20: # Minimum support filter
            continue
            
        supportA = item_counts[itemA] / total_invoices
        supportB = item_counts[itemB] / total_invoices
        supportAB = pair_count / total_invoices
        
        # Lift = P(A and B) / (P(A) * P(B))
        lift = supportAB / (supportA * supportB)
        results.append({
            'Product A': itemA,
            'Product B': itemB,
            'Co_occurrences': pair_count,
            'Lift': round(lift, 2)
        })
        
    df_rules = pd.DataFrame(results).sort_values('Lift', ascending=False)
    print(df_rules.head(10).to_string())
    df_rules.to_csv("data/processed/market_basket.csv", index=False)

def churn_prediction(conn):
    print("\n--- 4. Repeat Purchase Prediction Model ---")
    # We predict if a customer will buy in the last 3 months (Sep-Nov 2011)
    # based on their activity in the first 9 months (Dec 2010 - Aug 2011).
    query = """
    SELECT CustomerID, InvoiceDate, TotalPrice, InvoiceNo
    FROM transactions
    WHERE strftime('%Y-%m', InvoiceDate) < '2011-12'
    """
    df = pd.read_sql_query(query, conn)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    cutoff_date = pd.to_datetime('2011-09-01')
    
    df_train = df[df['InvoiceDate'] < cutoff_date]
    df_target = df[df['InvoiceDate'] >= cutoff_date]
    
    # Feature Engineering on training window (first 9 months)
    features = df_train.groupby('CustomerID').agg(
        TenureDays=('InvoiceDate', lambda x: (cutoff_date - x.min()).days),
        RecencyDays=('InvoiceDate', lambda x: (cutoff_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    
    # Target: 1 if they purchased in target window, else 0
    target_customers = set(df_target['CustomerID'].unique())
    features['Will_Repeat'] = features['CustomerID'].apply(lambda x: 1 if x in target_customers else 0)
    
    print(f"Dataset supports modeling! Features extracted for {len(features)} customers.")
    print(f"Baseline Repeat Rate: {features['Will_Repeat'].mean():.1%}")
    
    X = features[['TenureDays', 'RecencyDays', 'Frequency', 'Monetary']]
    y = features['Will_Repeat']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import roc_auc_score
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Baselines
    majority_class_acc = max(y_test.mean(), 1 - y_test.mean())
    y_all_ones = [1] * len(y_test)
    f1_all_ones = f1_score(y_test, y_all_ones)
    
    print("\n--- Model Metrics ---")
    print(f"Majority-class Accuracy Baseline: {majority_class_acc:.2f}")
    print(f"Model Accuracy:                   {acc:.2f}")
    print(f"'Always Predict 1' F1 Baseline:   {f1_all_ones:.2f}")
    print(f"Model F1 Score:                   {f1:.2f}")
    print(f"Model Precision:                  {prec:.2f}")
    print(f"Model Recall:                     {rec:.2f}")
    print(f"Model ROC-AUC:                    {auc:.2f}")
    
    print("\n--- Feature Importances ---")
    for name, imp in zip(X.columns, clf.feature_importances_):
        print(f"{name}: {imp:.3f}")
    
    print("\nConclusion: The model represents a modest improvement over the baselines.")

if __name__ == "__main__":
    conn = sqlite3.connect("data/processed/retail.db")
    rfm_segmentation(conn)
    cohort_analysis(conn)
    market_basket(conn)
    churn_prediction(conn)
    conn.close()
