# Business Insights & Recommendations

Based on the customer analytics pipeline applied to the UCI Online Retail dataset, here are the key insights and data-driven recommendations.

## 1. Which segments drive most of the revenue?
**Insight:** The Pareto principle is heavily evident in this dataset. The **Champions** segment (customers with high recency, frequency, and monetary scores) consists of just **809 customers** (18.7% of the customer base), but they generated **£5.35M in total revenue**—which is over 60% of the entire £8.74M revenue pool. Their Average Revenue Per Customer is **£6,620**, compared to just **£676** for Potential Loyalists (a ~9.8x difference). 

**Recommendation:** Suggested action to test: Create an exclusive "VIP / Platinum" tier specifically for the Champions segment. Test offering them early access to sales or free premium shipping to measure the impact on their retention.

## 2. Which customers are at risk of churning and worth a win-back campaign?
**Insight:** The **At Risk** segment currently holds **401 customers**. These are historically valuable customers who purchased frequently in the past but haven't bought anything recently. Together, they have historically generated **£659,000** (averaging £1,643 per customer). 

**Recommendation:** Suggested action to test: Launch an aggressive, personalized win-back email or SMS campaign specifically targeting these 401 customers. Test offering a high-value, one-time discount code (e.g., 20% off their next basket) to see if it reactivates their purchasing behavior.

## 3. How does retention change across cohorts?
**Insight:** Retention drops sharply in the first month across all cohorts. For the largest cohort (December 2010), only **36.5% of the 884 customers returned in Month 1**. However, the retention rate stabilizes and even rebounds over time. By Month 11 (November 2011), the retention rate for the December cohort climbed back up to **50.2%**, likely reflecting holiday-season purchasing. (Note: The December 2010 cohort may include pre-existing customers, artificially inflating its size and retention patterns compared to truly new cohorts).

**Recommendation:** Suggested action to test: Implement an automated "Month 1 Onboarding" email sequence. Test engaging new buyers with product education and a small "second purchase" incentive within their first 30 days to see if it flattens the initial churn curve.

## 4. Which products are worth cross-selling together?
**Insight:** The Market Basket Analysis revealed extremely strong co-occurrence patterns, mostly among colour variants bought in the same order. The strongest pair is the **ALARM CLOCK BAKELIKE GREEN** and **ALARM CLOCK BAKELIKE RED** (530 co-occurrences, Lift 14.10, meaning 14.1x as likely as chance). Another highly correlated pair is the **PAPER CHAIN KIT 50'S CHRISTMAS** and **PAPER CHAIN KIT VINTAGE CHRISTMAS** (451 co-occurrences, Lift 12.15).

**Recommendation:** Suggested action to test: Hardcode "Frequently Bought Together" bundles for these high-lift pairs directly on the product pages. Test showing these variants together to see if it increases the Average Order Value (AOV).
