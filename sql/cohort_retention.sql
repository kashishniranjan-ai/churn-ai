-- ==============================================================================
-- Monthly Cohort Retention Analysis Query
-- ==============================================================================
-- Tracks customer retention month-by-month. 
-- Note: The dataset ends on 2011-12-09. Since December 2011 is an incomplete 
-- month, we exclude any transactions from '2011-12' to prevent an artificial 
-- drop in retention metrics at the end of our heatmap.
-- ==============================================================================

WITH valid_transactions AS (
    -- Exclude incomplete month '2011-12'
    SELECT * 
    FROM transactions
    WHERE strftime('%Y-%m', InvoiceDate) < '2011-12'
),
ordered_purchases AS (
    SELECT 
        CustomerID,
        strftime('%Y-%m', InvoiceDate) AS ActivityMonth,
        ROW_NUMBER() OVER (PARTITION BY CustomerID ORDER BY InvoiceDate ASC) AS rn
    FROM valid_transactions
),
first_purchase AS (
    SELECT 
        CustomerID, 
        ActivityMonth AS CohortMonth
    FROM ordered_purchases
    WHERE rn = 1
),
transaction_month AS (
    SELECT DISTINCT
        CustomerID,
        strftime('%Y-%m', InvoiceDate) AS ActivityMonth
    FROM valid_transactions
),
cohort_activity AS (
    SELECT 
        t.CustomerID,
        f.CohortMonth,
        t.ActivityMonth
    FROM transaction_month t
    JOIN first_purchase f ON t.CustomerID = f.CustomerID
),
cohort_index AS (
    SELECT 
        CohortMonth,
        ActivityMonth,
        CustomerID,
        (CAST(substr(ActivityMonth, 1, 4) AS INTEGER) - CAST(substr(CohortMonth, 1, 4) AS INTEGER)) * 12 +
        (CAST(substr(ActivityMonth, 6, 2) AS INTEGER) - CAST(substr(CohortMonth, 6, 2) AS INTEGER)) AS MonthIndex
    FROM cohort_activity
),
cohort_counts AS (
    SELECT 
        CohortMonth,
        MonthIndex,
        COUNT(DISTINCT CustomerID) AS CustomerCount
    FROM cohort_index
    GROUP BY CohortMonth, MonthIndex
)
SELECT 
    CohortMonth,
    MonthIndex,
    CustomerCount,
    -- FIRST_VALUE gets the initial size of the cohort (MonthIndex = 0)
    FIRST_VALUE(CustomerCount) OVER (
        PARTITION BY CohortMonth 
        ORDER BY MonthIndex ASC
    ) AS InitialSize,
    -- LAG gets the count from the immediately preceding active month
    LAG(CustomerCount) OVER (
        PARTITION BY CohortMonth 
        ORDER BY MonthIndex ASC
    ) AS PrevMonthCount,
    -- Absolute Retention percentage compared to Month 0
    ROUND(
        CAST(CustomerCount AS FLOAT) / 
        FIRST_VALUE(CustomerCount) OVER (PARTITION BY CohortMonth ORDER BY MonthIndex ASC) * 100, 
    2) AS RetentionRate,
    -- Note on Active Counts: Customers don't buy every month, so active count 
    -- can temporarily rise (e.g. going from Month 2 -> Month 3). This ratio 
    -- reflects activity momentum relative to the previous month, not strict 
    -- continuous retention.
    ROUND(
        CAST(CustomerCount AS FLOAT) / 
        LAG(CustomerCount) OVER (PARTITION BY CohortMonth ORDER BY MonthIndex ASC) * 100, 
    2) AS MoM_ActiveRatio
FROM cohort_counts
ORDER BY CohortMonth, MonthIndex;
