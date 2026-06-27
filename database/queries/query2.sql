SELECT
    contract,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(AVG(CAST(churn AS FLOAT)) * 100, 2) AS churn_rate
FROM customers
GROUP BY contract
ORDER BY churn_rate DESC;