-- ==============================================================================
-- RFM (Recency, Frequency, Monetary) Analysis Query
-- ==============================================================================
-- Computes the RFM metrics for each customer.
--
-- Note on Frequency: Over 34% of customers have exactly 1 purchase. 
-- Using percentile-based distribution (like NTILE or CUME_DIST) causes 
-- Bucket 1 to be completely empty because identical values must fall into 
-- the same bucket. Instead, we use fixed thresholds for Frequency to 
-- guarantee meaningful buckets:
-- 1 = 1 purchase
-- 2 = 2 purchases
-- 3 = 3-4 purchases
-- 4 = 5-9 purchases
-- 5 = 10+ purchases
-- ==============================================================================

WITH raw_rfm AS (
    SELECT 
        CustomerID,
        CAST(julianday('2011-12-10') - julianday(MAX(InvoiceDate)) AS INTEGER) AS Recency,
        COUNT(DISTINCT InvoiceNo) AS Frequency,
        ROUND(SUM(TotalPrice), 2) AS Monetary
    FROM transactions
    GROUP BY CustomerID
),
percentiles AS (
    SELECT 
        CustomerID,
        Recency,
        Frequency,
        Monetary,
        CUME_DIST() OVER (ORDER BY Recency ASC) AS pct_R,
        CUME_DIST() OVER (ORDER BY Monetary ASC) AS pct_M
    FROM raw_rfm
),
rfm_scores AS (
    SELECT 
        CustomerID,
        Recency,
        Frequency,
        Monetary,
        -- Recency: Lower is better, so the bottom 20% gets a 5.
        CASE 
            WHEN pct_R <= 0.2 THEN 5
            WHEN pct_R <= 0.4 THEN 4
            WHEN pct_R <= 0.6 THEN 3
            WHEN pct_R <= 0.8 THEN 2
            ELSE 1 
        END AS R_Score,
        -- Frequency: Use fixed thresholds due to heavy ties on low values.
        CASE 
            WHEN Frequency = 1 THEN 1
            WHEN Frequency = 2 THEN 2
            WHEN Frequency BETWEEN 3 AND 4 THEN 3
            WHEN Frequency BETWEEN 5 AND 9 THEN 4
            ELSE 5 
        END AS F_Score,
        -- Monetary: Higher is better, so the bottom 20% gets a 1.
        CASE 
            WHEN pct_M <= 0.2 THEN 1
            WHEN pct_M <= 0.4 THEN 2
            WHEN pct_M <= 0.6 THEN 3
            WHEN pct_M <= 0.8 THEN 4
            ELSE 5 
        END AS M_Score
    FROM percentiles
)
SELECT 
    CustomerID,
    Recency,
    Frequency,
    Monetary,
    R_Score,
    F_Score,
    M_Score,
    CAST(R_Score AS TEXT) || CAST(F_Score AS TEXT) || CAST(M_Score AS TEXT) AS RFM_Score
FROM rfm_scores
ORDER BY RFM_Score DESC, Monetary DESC;
