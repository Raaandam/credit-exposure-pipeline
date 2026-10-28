"""Small, hand-built pandas fixtures: every injected defect is visible by reading the data."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

BD = date(2026, 1, 7)


@pytest.fixture
def bd() -> date:
    return BD


@pytest.fixture
def raw_datasets() -> dict:
    cps = pd.DataFrame({
        "counterparty_id": ["CP000001", "CP000002", "CP000003", "CP000004", "CP000004"],  # CP000004 duplicated
        "counterparty_name": ["Alpine Holdings AG", "Nordic Capital SA", "Atlas Energy plc", "Orion Pharma Inc", "Orion Pharma Inc"],
        "lei": ["ABCDEFGHIJKLMNOPQR12", "BCDEFGHIJKLMNOPQRS34", "CDEFGHIJKLMNOPQRST56", "DEFGHIJKLMNOPQRSTU78", "DEFGHIJKLMNOPQRSTU78"],
        "group_id": ["GRP00001", "GRP00001", "GRP00002", "GRP00003", "GRP00003"],
        "ultimate_parent_id": ["CP000001", "CP000001", "CP000003", "CP000004", "CP000004"],
        "country": ["CH", "US", "GBR", "DE", "DE"],                              # GBR not ISO-2
        "sector": ["BANKS", "FIN-01", "ENERGY", "HEALTHCARE", "HEALTHCARE"],      # FIN-01 legacy code
        "internal_rating": ["A", "BBB", np.nan, "Baa2", "Baa2"],                  # missing + agency code
        "as_of_date": [BD.isoformat()] * 5,
    })
    fac = pd.DataFrame({
        "facility_id": ["FAC0000001", "FAC0000002", "FAC0000003", "FAC0000004", "FAC0000005", "FAC0000005"],
        "counterparty_id": ["CP000001", "CP000002", "CP000003", "CP999999", "CP000004", "CP000004"],  # orphan, dup
        "facility_type": ["TERM_LOAN", "REVOLVER", "GUARANTEE", "REVOLVER", "OVERDRAFT", "OVERDRAFT"],
        "currency": ["CHF", "USD", "GBX", "EUR", np.nan, np.nan],
        "limit_amount": ["1000000", "500000", "200000000000", "300000", "40000", "40000"],  # GBX pence -> too large
        "drawn_amount": ["1000000", "200000", "100000", "350000", "10000", "10000"],         # FAC4 overdrawn
        "maturity_date": ["2030-01-01"] * 6,
        "source_system": ["LOAN_SYS_EMEA"] * 6,
        "as_of_date": [BD.isoformat()] * 6,
    })
    col = pd.DataFrame({
        "collateral_id": ["COL0000001", "COL0000002", "COL0000003"],
        "facility_id": ["FAC0000001", "FAC0000002", "FAC9000000"],  # orphan facility
        "collateral_type": ["CASH", "REAL_ESTATE", "EQUITY"],
        "currency": ["CHF", "USD", "CHF"],
        "market_value": [400000.0, np.nan, 50000.0],
        "haircut": [0.0, 0.35, 25.0],  # percent instead of decimal
        "as_of_date": [BD.isoformat()] * 3,
    })
    rc = pd.DataFrame({
        "counterparty_id": ["CP000001", "CP000003"],
        "previous_rating": ["AA", "BBB"],
        "new_rating": ["A", "CCC"],  # 3-notch downgrade
        "effective_date": [BD.isoformat(), "2026-01-06"],  # one stale row
        "change_reason": ["ANNUAL_REVIEW", "CREDIT_EVENT"],
        "rating_source": ["INTERNAL"] * 2,
    })
    fx = pd.DataFrame({"currency": ["CHF", "USD", "EUR", "GBP"], "rate_to_chf": ["1.0", "0.9", "0.95", "1.1"],
                       "as_of_date": [BD.isoformat()] * 4})
    return {"raw.counterparties": cps, "raw.facilities": fac, "raw.collateral": col, "raw.rating_changes": rc, "raw.fx_rates": fx}


@pytest.fixture
def manifest(raw_datasets) -> dict:
    fac = raw_datasets["raw.facilities"]
    return {
        "counterparties": {"record_count": 5, "business_date": BD.isoformat()},
        "facilities": {"record_count": 7,  # one row lost in transfer
                       "control_total_drawn": float(pd.to_numeric(fac["drawn_amount"]).sum()),
                       "control_total_limit": float(pd.to_numeric(fac["limit_amount"]).sum()) * 1.01},
        "collateral": {"record_count": 3, "control_total_market_value": 450000.0},
        "rating_changes": {"record_count": 2},
        "fx_rates": {"record_count": 4},
    }
