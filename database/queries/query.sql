SELECT
    COUNT(*) AS total_customers,
    SUM(churn) AS churned,
    ROUND(AVG(CAST(churn AS FLOAT)) * 100, 2) AS churn_rate
FROM customers;