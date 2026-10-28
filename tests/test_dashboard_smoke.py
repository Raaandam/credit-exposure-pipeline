"""Render each Streamlit page headlessly against a small generated SQLite DB."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASH = Path(__file__).resolve().parents[1] / "dashboard"


@pytest.fixture(scope="module")
def db_url(tmp_path_factory):
    from exposure_pipeline.dq.backends.pandas_backend import PandasBackend
    from exposure_pipeline.dq.engine import DQEngine
    from exposure_pipeline.dq.rules import load_rules
    from exposure_pipeline.generator.generate_feeds import GeneratorConfig, business_days, generate
    from exposure_pipeline.local_pipeline import run_business_date
    from exposure_pipeline.storage.db import ensure_schema, get_engine

    out = tmp_path_factory.mktemp("dash")
    generate(GeneratorConfig(out_dir=out, n_counterparties=150, n_days=2, seed=3))
    url = f"sqlite:///{(out / 'dash.db').as_posix()}"
    engine = get_engine(url)
    ensure_schema(engine)
    dq = DQEngine(load_rules(), PandasBackend())
    for d in business_days(date(2026, 1, 5), 2):
        run_business_date(d, engine, out / "raw", dq)
    return url


@pytest.mark.parametrize("page", ["app.py", "pages/1_Exposure.py", "pages/2_Triage_Review.py"])
def test_page_renders(page, db_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", db_url)
    at = AppTest.from_file(str(DASH / page), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
