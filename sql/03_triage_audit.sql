-- Full audit trail of every LLM triage call and the human decision on it.
-- Append-only for the model part; reviewer columns are written exactly once
-- (application enforces "UPDATE ... WHERE reviewer_decision IS NULL", the trigger backs it up).
CREATE TABLE IF NOT EXISTS triage_audit (
    audit_id            VARCHAR(36)  PRIMARY KEY,
    result_id           VARCHAR(80)  NOT NULL,           -- dq_results.result_id (not FK: results are replaced on rerun)
    business_date       DATE,
    rule_id             VARCHAR(40),
    created_at          TIMESTAMP    NOT NULL,
    model_requested     VARCHAR(64),
    model_version       VARCHAR(128),                    -- as returned by the API, e.g. dated snapshot
    prompt_version      VARCHAR(32),
    temperature         DOUBLE PRECISION,
    prompt_hash         VARCHAR(64),                     -- sha256 of model + prompt version + rendered prompt
    prompt_text         TEXT,                            -- full (masked) prompt as sent
    output_text         TEXT,                            -- raw model output
    llm_category        VARCHAR(40),                     -- what the model said
    suggested_category  VARCHAR(40),                     -- after guardrails (may be needs_manual_review)
    confidence          DOUBLE PRECISION,
    root_cause_note     TEXT,
    suggested_fix       TEXT,
    fallback_reason     TEXT,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    latency_ms          DOUBLE PRECISION,
    cost_usd            DOUBLE PRECISION,
    reviewer            VARCHAR(100),
    reviewer_decision   VARCHAR(16) CHECK (reviewer_decision IN ('approved', 'edited', 'rejected')),
    reviewed_at         TIMESTAMP,
    final_category      VARCHAR(40),
    final_note          TEXT
);
CREATE INDEX IF NOT EXISTS ix_triage_audit_result ON triage_audit (result_id);
CREATE INDEX IF NOT EXISTS ix_triage_audit_pending ON triage_audit (reviewer_decision) WHERE reviewer_decision IS NULL;

-- Guard: model output columns are immutable and a review cannot be overwritten.
CREATE OR REPLACE FUNCTION triage_audit_guard() RETURNS trigger AS $$
BEGIN
    IF OLD.reviewer_decision IS NOT NULL THEN
        RAISE EXCEPTION 'triage_audit % already reviewed', OLD.audit_id;
    END IF;
    IF NEW.prompt_text IS DISTINCT FROM OLD.prompt_text
       OR NEW.output_text IS DISTINCT FROM OLD.output_text
       OR NEW.llm_category IS DISTINCT FROM OLD.llm_category
       OR NEW.confidence IS DISTINCT FROM OLD.confidence THEN
        RAISE EXCEPTION 'model output columns of triage_audit are immutable';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_triage_audit_guard ON triage_audit;
CREATE TRIGGER trg_triage_audit_guard BEFORE UPDATE ON triage_audit
    FOR EACH ROW EXECUTE FUNCTION triage_audit_guard();
