SELECT
    CASE
        WHEN monthlycharges < -0.5 THEN 'Low'
        WHEN monthlycharges < 0.5 THEN 'Medium'
        ELSE 'High'
    END AS charge_level,
    ROUND(AVG(CAST(churn AS FLOAT)) * 100, 2) AS churn_rate
FROM customers
GROUP BY CASE ... END;