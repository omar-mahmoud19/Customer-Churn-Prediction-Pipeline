-- ============================================================
-- churn_analysis.sql
-- Role: SQL Engineer
-- Purpose: Business intelligence queries for churn analysis.
-- ============================================================


-- ── 1. Overall churn rate ─────────────────────────────────────
-- KPI: What percentage of customers have churned?
SELECT
    COUNT(*)                                       AS total_customers,
    SUM(churn)                                     AS churned_customers,
    ROUND(AVG(churn::NUMERIC) * 100, 2)            AS churn_rate_pct
FROM customers;


-- ── 2. Churn rate by contract type ───────────────────────────
-- Insight: Month-to-month customers churn far more.
SELECT
    s.contract,
    COUNT(c.customer_id)                           AS total_customers,
    SUM(c.churn)                                   AS churned,
    ROUND(AVG(c.churn::NUMERIC) * 100, 2)          AS churn_rate_pct
FROM customers c
JOIN customer_subscriptions s USING (customer_id)
GROUP BY s.contract
ORDER BY churn_rate_pct DESC;


-- ── 3. Churn rate by tenure bucket ───────────────────────────
-- Insight: New customers (< 12 months) are at highest risk.
SELECT
    CASE
        WHEN s.tenure < 12   THEN '0–12 months'
        WHEN s.tenure < 24   THEN '12–24 months'
        WHEN s.tenure < 48   THEN '24–48 months'
        ELSE                      '48+ months'
    END                                            AS tenure_bucket,
    COUNT(c.customer_id)                           AS total_customers,
    SUM(c.churn)                                   AS churned,
    ROUND(AVG(c.churn::NUMERIC) * 100, 2)          AS churn_rate_pct
FROM customers c
JOIN customer_subscriptions s USING (customer_id)
GROUP BY tenure_bucket
ORDER BY churn_rate_pct DESC;


-- ── 4. High-risk customer segmentation ───────────────────────
-- Segment: Customers predicted to churn with probability > 70%
SELECT
    c.customer_id,
    c.age,
    c.annual_income,
    s.contract,
    s.monthly_charges,
    s.tenure,
    p.churn_probability,
    e.num_complaints,
    e.customer_satisfaction
FROM customers c
JOIN customer_subscriptions s   USING (customer_id)
JOIN churn_predictions p        USING (customer_id)
LEFT JOIN customer_service_events e USING (customer_id)
WHERE p.churn_probability > 0.70
ORDER BY p.churn_probability DESC;


-- ── 5. High complaint customers ──────────────────────────────
-- Operational: Customers needing immediate intervention.
SELECT
    c.customer_id,
    e.num_complaints,
    e.num_service_calls,
    e.customer_satisfaction,
    e.late_payments,
    p.churn_probability
FROM customers c
JOIN customer_service_events e  USING (customer_id)
LEFT JOIN churn_predictions p   USING (customer_id)
WHERE e.num_complaints >= 3
ORDER BY e.num_complaints DESC, p.churn_probability DESC;


-- ── 6. Revenue at risk from predicted churners ───────────────
-- Finance: Monthly revenue exposed to churn.
SELECT
    ROUND(SUM(s.monthly_charges), 2)               AS monthly_revenue_at_risk,
    COUNT(p.customer_id)                           AS predicted_churners,
    ROUND(AVG(s.monthly_charges), 2)               AS avg_monthly_charge
FROM churn_predictions p
JOIN customer_subscriptions s USING (customer_id)
WHERE p.predicted_churn = 1;


-- ── 7. Churn rate by number of services ──────────────────────
-- Insight: Customers with fewer services churn more.
SELECT
    s.num_services,
    COUNT(c.customer_id)                           AS total_customers,
    SUM(c.churn)                                   AS churned,
    ROUND(AVG(c.churn::NUMERIC) * 100, 2)          AS churn_rate_pct
FROM customers c
JOIN customer_subscriptions s USING (customer_id)
GROUP BY s.num_services
ORDER BY s.num_services;


-- ── 8. Customer satisfaction vs churn ────────────────────────
SELECT
    CASE
        WHEN e.customer_satisfaction < 2  THEN 'Very Dissatisfied (< 2)'
        WHEN e.customer_satisfaction < 3  THEN 'Dissatisfied (2–3)'
        WHEN e.customer_satisfaction < 4  THEN 'Neutral (3–4)'
        ELSE                                   'Satisfied (4–5)'
    END                                            AS satisfaction_band,
    COUNT(c.customer_id)                           AS total_customers,
    ROUND(AVG(c.churn::NUMERIC) * 100, 2)          AS churn_rate_pct
FROM customers c
JOIN customer_service_events e USING (customer_id)
GROUP BY satisfaction_band
ORDER BY churn_rate_pct DESC;


-- ── 9. Monthly cohort churn analysis ─────────────────────────
SELECT
    DATE_TRUNC('month', c.signup_date)::DATE       AS signup_cohort,
    COUNT(c.customer_id)                           AS cohort_size,
    SUM(c.churn)                                   AS churned,
    ROUND(AVG(c.churn::NUMERIC) * 100, 2)          AS churn_rate_pct
FROM customers c
GROUP BY signup_cohort
ORDER BY signup_cohort DESC;


-- ── 10. Payment method churn breakdown ───────────────────────
SELECT
    s.payment_method,
    COUNT(c.customer_id)                           AS total_customers,
    SUM(c.churn)                                   AS churned,
    ROUND(AVG(c.churn::NUMERIC) * 100, 2)          AS churn_rate_pct
FROM customers c
JOIN customer_subscriptions s USING (customer_id)
GROUP BY s.payment_method
ORDER BY churn_rate_pct DESC;
