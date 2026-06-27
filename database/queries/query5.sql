SELECT
    CASE
        WHEN tenure < -0.5 THEN 'New (< 1yr)'
        WHEN tenure < 0.5 THEN 'Mid (1-3yr)'
        ELSE 'Loyal (3yr+)'
    END AS tenure_group,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(AVG(CAST(churn AS FLOAT)) * 100, 2) AS churn_rate
FROM customers
GROUP BY CASE ... END
ORDER BY churn_rate DESC;