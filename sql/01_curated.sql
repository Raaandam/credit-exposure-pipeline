-- Curated layer + reporting mart. All tables are partitioned logically by business_date;
-- loads are "delete partition + insert" in one transaction (idempotent reruns).
-- Mirrors src/exposure_pipeline/storage/tables.py.

CREATE TABLE IF NOT EXISTS cur_counterparty (
    business_date       DATE         NOT NULL,
    counterparty_id     VARCHAR(32)  NOT NULL,
    counterparty_name   VARCHAR(200),
    lei                 VARCHAR(20),
    group_id            VARCHAR(32),
    ultimate_parent_id  VARCHAR(32),
    country             VARCHAR(32),
    sector              VARCHAR(64),
    internal_rating     VARCHAR(8),
    rating_band         VARCHAR(16),
    PRIMARY KEY (business_date, counterparty_id)
);

CREATE TABLE IF NOT EXISTS cur_facility_exposure (
    business_date     DATE         NOT NULL,
    facility_id       VARCHAR(32)  NOT NULL,
    counterparty_id   VARCHAR(32),
    group_id          VARCHAR(32),
    sector            VARCHAR(64),
    rating_band       VARCHAR(16),
    facility_type     VARCHAR(32),
    currency          VARCHAR(8),
    source_system     VARCHAR(32),
    limit_amount      DOUBLE PRECISION,
    drawn_amount      DOUBLE PRECISION,
    fx_rate_to_chf    DOUBLE PRECISION,
    limit_chf         DOUBLE PRECISION,
    drawn_chf         DOUBLE PRECISION,
    undrawn_chf       DOUBLE PRECISION,
    ccf               DOUBLE PRECISION,
    ead_chf           DOUBLE PRECISION,   -- drawn + CCF * undrawn, in CHF
    collateral_chf    DOUBLE PRECISION,   -- recognised collateral, capped at EAD
    net_exposure_chf  DOUBLE PRECISION,   -- collateral-adjusted exposure
    maturity_date     VARCHAR(10),
    is_orphan         BOOLEAN,
    PRIMARY KEY (business_date, facility_id)
);
CREATE INDEX IF NOT EXISTS ix_cfe_group ON cur_facility_exposure (business_date, group_id);

CREATE TABLE IF NOT EXISTS cur_collateral (
    business_date       DATE         NOT NULL,
    collateral_id       VARCHAR(32)  NOT NULL,
    facility_id         VARCHAR(32),
    collateral_type     VARCHAR(32),
    currency            VARCHAR(8),
    market_value        DOUBLE PRECISION,
    haircut             DOUBLE PRECISION,
    adjusted_value_chf  DOUBLE PRECISION,
    PRIMARY KEY (business_date, collateral_id)
);

CREATE TABLE IF NOT EXISTS cur_rating_change (
    business_date     DATE         NOT NULL,
    counterparty_id   VARCHAR(32)  NOT NULL,
    previous_rating   VARCHAR(8),
    new_rating        VARCHAR(8),
    notch_change      DOUBLE PRECISION,
    notch_change_abs  DOUBLE PRECISION,
    effective_date    VARCHAR(10),
    change_reason     VARCHAR(64),
    PRIMARY KEY (business_date, counterparty_id)
);

-- Long-format mart: dimension in (COUNTERPARTY_GROUP, SECTOR, RATING_BAND).
-- Feeds credit risk reporting, limit monitoring (by group) and RWA input (by rating band).
CREATE TABLE IF NOT EXISTS mart_exposure_aggregate (
    business_date       DATE         NOT NULL,
    dimension           VARCHAR(32)  NOT NULL,
    dimension_value     VARCHAR(64)  NOT NULL,
    ead_chf             DOUBLE PRECISION,
    collateral_chf      DOUBLE PRECISION,
    net_exposure_chf    DOUBLE PRECISION,
    facility_count      INTEGER,
    counterparty_count  INTEGER,
    PRIMARY KEY (business_date, dimension, dimension_value)
);

-- One row per feed per business date: what arrived, from which extract.
CREATE TABLE IF NOT EXISTS load_audit (
    business_date           DATE         NOT NULL,
    feed                    VARCHAR(32)  NOT NULL,
    run_id                  VARCHAR(64)  NOT NULL,
    status                  VARCHAR(16)  NOT NULL,   -- SUCCESS | MISSING | FAILED
    row_count               INTEGER,
    source_path             TEXT,
    source_system           VARCHAR(64),
    manifest_business_date  VARCHAR(10),
    manifest_record_count   INTEGER,
    extract_timestamp       VARCHAR(32),
    loaded_at               TIMESTAMP    NOT NULL,
    PRIMARY KEY (business_date, feed)
);
