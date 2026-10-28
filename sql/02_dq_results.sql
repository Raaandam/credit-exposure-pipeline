-- One row per (business_date, rule). result_id = '<business_date>:<rule_id>' so reruns
-- replace rows and triage_audit keeps pointing at the same logical break.
CREATE TABLE IF NOT EXISTS dq_results (
    result_id           VARCHAR(80)  PRIMARY KEY,
    run_id              VARCHAR(64)  NOT NULL,
    business_date       DATE         NOT NULL,
    rule_id             VARCHAR(40)  NOT NULL,
    rule_name           VARCHAR(200),
    feed                VARCHAR(32),
    dataset             VARCHAR(64),
    dimension           VARCHAR(32),
    check_type          VARCHAR(40),
    severity            VARCHAR(16),
    status              VARCHAR(8)   NOT NULL CHECK (status IN ('PASS', 'FAIL', 'ERROR')),
    total_count         INTEGER,
    failed_count        INTEGER,
    failure_rate        DOUBLE PRECISION,
    observed_value      TEXT,
    expected_value      TEXT,
    message             TEXT,
    failed_keys         JSONB,       -- sample of failing business keys
    sample_records      JSONB,       -- up to N failing records (unmasked; masking happens before any LLM call)
    executed_at         TIMESTAMP    NOT NULL,
    -- Written only by an approved human review, never directly by the LLM:
    triage_status       VARCHAR(16)  DEFAULT 'untriaged',
    triage_category     VARCHAR(40),
    triage_note         TEXT,
    triage_reviewed_by  VARCHAR(100),
    triage_reviewed_at  TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_dq_results_date_status ON dq_results (business_date, status);
CREATE INDEX IF NOT EXISTS ix_dq_results_rule ON dq_results (rule_id, business_date);
