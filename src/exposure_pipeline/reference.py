"""Reference data and conventions shared by the generator, pandas/Spark transforms and DQ rules.

Keeping these in one module means the generator, the legacy batch, the Spark job and the
DQ rules cannot silently drift apart on, e.g., the rating scale or CCF table.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

# Internal rating master scale (letter grades, best to worst). Rank 0 = best.
RATING_SCALE: tuple[str, ...] = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D")
RATING_RANK: dict[str, int] = {r: i for i, r in enumerate(RATING_SCALE)}

# Rating bands used for reporting / RWA input. UNRATED is an explicit bucket so that
# exposure with a missing or unmapped rating is visible rather than dropped.
RATING_BAND: dict[str, str] = {
    "AAA": "IG_HIGH",
    "AA": "IG_HIGH",
    "A": "IG_HIGH",
    "BBB": "IG_LOW",
    "BB": "SUB_IG",
    "B": "SUB_IG",
    "CCC": "WATCHLIST",
    "D": "DEFAULT",
}
UNRATED_BAND = "UNRATED"
UNMAPPED = "UNMAPPED"

SECTORS: tuple[str, ...] = (
    "BANKS",
    "INSURANCE",
    "ASSET_MANAGEMENT",
    "SOVEREIGN",
    "ENERGY",
    "REAL_ESTATE",
    "INDUSTRIALS",
    "TECHNOLOGY",
    "HEALTHCARE",
    "CONSUMER",
    "UTILITIES",
    "TRANSPORT",
)

REPORTING_CCY = "CHF"
# Indicative starting FX rates (units of CHF per 1 unit of currency). The generator adds a
# small daily random walk; values are synthetic.
BASE_FX_TO_CHF: dict[str, float] = {
    "CHF": 1.0,
    "USD": 0.88,
    "EUR": 0.94,
    "GBP": 1.11,
    "JPY": 0.0059,
    "SGD": 0.66,
    "HKD": 0.113,
    "CAD": 0.64,
    "AUD": 0.58,
}

# Illustrative credit conversion factors applied to the undrawn amount. Loosely inspired by
# the Basel standardised approach; this is NOT a regulatory EAD calculation.
FACILITY_CCF: dict[str, float] = {
    "TERM_LOAN": 1.0,
    "REVOLVER": 0.4,
    "TRADE_FINANCE": 0.2,
    "GUARANTEE": 0.5,
    "OVERDRAFT": 0.1,
}

# Typical supervisory-style haircuts by collateral type (generator uses them as a base).
COLLATERAL_HAIRCUT: dict[str, float] = {
    "CASH": 0.0,
    "GOVT_BOND": 0.04,
    "CORP_BOND": 0.12,
    "EQUITY": 0.25,
    "REAL_ESTATE": 0.35,
    "RECEIVABLES": 0.40,
}

FEED_FILES: dict[str, str] = {
    "counterparties": "counterparties.csv",
    "facilities": "facilities.csv",
    "collateral": "collateral.json",  # newline-delimited JSON
    "rating_changes": "rating_changes.csv",
    "fx_rates": "fx_rates.csv",
}
MANIFEST_FILE = "manifest.json"

# Business keys per feed. Used for dedup, DQ failing-key samples and ground-truth labels.
FEED_KEYS: dict[str, list[str]] = {
    "counterparties": ["counterparty_id"],
    "facilities": ["facility_id"],
    "collateral": ["collateral_id"],
    "rating_changes": ["counterparty_id"],
    "fx_rates": ["currency"],
}

AGGREGATION_DIMENSIONS: dict[str, str] = {
    "COUNTERPARTY_GROUP": "group_id",
    "SECTOR": "sector",
    "RATING_BAND": "rating_band",
}

# Triage taxonomy (shared by the generator labels, the LLM schema and the evaluation).
BREAK_CATEGORIES: tuple[str, ...] = (
    "missing_source_data",
    "mapping_error",
    "stale_feed",
    "duplicate",
    "unit_or_fx_mismatch",
    "genuine_business_change",
)


def partition_dir(root: Path, business_date: date | str) -> Path:
    """Hive-style partition folder, e.g. data/raw/business_date=2026-01-05."""
    bd = business_date if isinstance(business_date, str) else business_date.isoformat()
    return Path(root) / f"business_date={bd}"
