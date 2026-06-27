SELECT TOP 100
    customer_id, num_complaints,
    customer_satisfaction, contract, churn
FROM customers
WHERE num_complaints > 0
  AND customer_satisfaction < 0
  AND churn = 0
ORDER BY num_complaints DESC;