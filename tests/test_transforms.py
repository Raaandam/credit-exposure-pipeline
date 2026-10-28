"""Curation / aggregation business rules (pandas reference implementation)."""
from __future__ import annotations

import pytest

from exposure_pipeline.transform.pandas_impl import curate_all


@pytest.fixture
def curated(raw_datasets, bd):
    return curate_all(raw_datasets, bd)


def test_dedupe_keeps_one_row_per_key(curated):
    assert curated["curated.counterparty"]["counterparty_id"].is_unique
    assert curated["curated.facility_exposure"]["facility_id"].is_unique
    assert len(curated["curated.facility_exposure"]) == 5


def test_ead_formula_and_fx(curated):
    fe = curated["curated.facility_exposure"].set_index("facility_id")
    # TERM_LOAN fully drawn in CHF: EAD = drawn = 1m
    assert fe.loc["FAC0000001", "ead_chf"] == pytest.approx(1_000_000)
    # REVOLVER USD: (200k + 0.4 * 300k) * 0.9
    assert fe.loc["FAC0000002", "ead_chf"] == pytest.approx((200_000 + 0.4 * 300_000) * 0.9)
    # GBX has no FX rate -> no EAD rather than a wrong one
    assert fe.loc["FAC0000003", "ead_chf"] != fe.loc["FAC0000003", "ead_chf"]  # NaN


def test_collateral_capped_at_ead_and_haircut_clipped(curated):
    fe = curated["curated.facility_exposure"].set_index("facility_id")
    assert fe.loc["FAC0000001", "collateral_chf"] == pytest.approx(400_000)
    assert fe.loc["FAC0000001", "net_exposure_chf"] == pytest.approx(600_000)
    col = curated["curated.collateral"].set_index("collateral_id")
    assert col.loc["COL0000003", "adjusted_value_chf"] == 0  # haircut 25 (percent) clipped to 1


def test_orphans_and_unrated_are_bucketed_not_dropped(curated):
    fe = curated["curated.facility_exposure"].set_index("facility_id")
    assert bool(fe.loc["FAC0000004", "is_orphan"]) and fe.loc["FAC0000004", "group_id"] == "UNMAPPED"
    cps = curated["curated.counterparty"].set_index("counterparty_id")
    assert cps.loc["CP000003", "rating_band"] == "UNRATED"
    assert cps.loc["CP000004", "rating_band"] == "UNRATED"  # agency code not mapped
    assert cps.loc["CP000002", "sector"] == "UNMAPPED"      # legacy sector code


def test_every_dimension_sums_to_the_same_total(curated):
    agg = curated["mart.exposure_aggregate"]
    total = curated["curated.facility_exposure"]["ead_chf"].sum()
    for dim, g in agg.groupby("dimension"):
        assert g["ead_chf"].sum() == pytest.approx(total), dim


def test_notch_change(curated):
    rc = curated["curated.rating_change"].set_index("counterparty_id")
    assert rc.loc["CP000003", "notch_change"] == 3   # BBB -> CCC downgrade
    assert rc.loc["CP000001", "notch_change"] == 1   # AA -> A
