-- Airflow metadata lives in its own database on the same Postgres instance (dev convenience;
-- in production it would be a separate managed instance).
CREATE DATABASE airflow;
