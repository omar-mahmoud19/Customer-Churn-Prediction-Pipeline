-- ============================================================
-- create_tables.sql
-- Role: SQL Engineer
-- Purpose: Define the database schema for the churn platform.
-- ============================================================

-- ── Drop tables if they exist (safe re-run) ──────────────────
DROP TABLE IF EXISTS churn_predictions;
DROP TABLE IF EXISTS customer_service_events;
DROP TABLE IF EXISTS customer_subscriptions;
DROP TABLE IF EXISTS customers;

-- ── Core customer table ───────────────────────────────────────
CREATE TABLE customers (
    customer_id             VARCHAR(20)   PRIMARY KEY,
    signup_date             TIMESTAMP,
    age                     INTEGER,
    gender                  VARCHAR(10),
    annual_income           NUMERIC(12,2),
    education               VARCHAR(30),
    marital_status          VARCHAR(20),
    dependents              INTEGER       DEFAULT 0,
    senior_citizen          SMALLINT      DEFAULT 0,
    credit_score            NUMERIC(6,2),
    days_since_last_interaction INTEGER,
    churn                   SMALLINT      DEFAULT 0,
    created_at              TIMESTAMP     DEFAULT CURRENT_TIMESTAMP
);

-- ── Subscription / contract details ──────────────────────────
CREATE TABLE customer_subscriptions (
    subscription_id         SERIAL        PRIMARY KEY,
    customer_id             VARCHAR(20)   NOT NULL REFERENCES customers(customer_id),
    tenure                  INTEGER,
    contract                VARCHAR(20),          -- Month-to-Month, One Year, Two Year
    payment_method          VARCHAR(40),
    paperless_billing       VARCHAR(5),
    monthly_charges         NUMERIC(8,2),
    total_charges           NUMERIC(12,2),
    num_services            INTEGER,
    has_phone_service       SMALLINT      DEFAULT 0,
    has_internet_service    SMALLINT      DEFAULT 0,
    has_online_security     SMALLINT      DEFAULT 0,
    has_online_backup       SMALLINT      DEFAULT 0,
    has_device_protection   SMALLINT      DEFAULT 0,
    has_tech_support        SMALLINT      DEFAULT 0,
    has_streaming_tv        SMALLINT      DEFAULT 0,
    has_streaming_movies    SMALLINT      DEFAULT 0,
    avg_monthly_gb          NUMERIC(8,2),
    CONSTRAINT fk_sub_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ── Service & payment events ──────────────────────────────────
CREATE TABLE customer_service_events (
    event_id                SERIAL        PRIMARY KEY,
    customer_id             VARCHAR(20)   NOT NULL,
    customer_satisfaction   NUMERIC(4,2),
    num_complaints          INTEGER       DEFAULT 0,
    num_service_calls       INTEGER       DEFAULT 0,
    late_payments           INTEGER       DEFAULT 0,
    recorded_at             TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_evt_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ── ML prediction results ─────────────────────────────────────
CREATE TABLE churn_predictions (
    prediction_id           SERIAL        PRIMARY KEY,
    customer_id             VARCHAR(20)   NOT NULL,
    predicted_churn         SMALLINT      NOT NULL,
    churn_probability       NUMERIC(6,4)  NOT NULL,
    model_version           VARCHAR(50),
    predicted_at            TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_pred_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ── Indexes for common query patterns ────────────────────────
CREATE INDEX idx_customers_churn           ON customers(churn);
CREATE INDEX idx_customers_contract        ON customer_subscriptions(contract);
CREATE INDEX idx_predictions_probability   ON churn_predictions(churn_probability DESC);
CREATE INDEX idx_events_complaints         ON customer_service_events(num_complaints DESC);
