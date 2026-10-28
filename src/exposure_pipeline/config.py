"""Environment-driven settings.

Design choice: every environment-specific value (storage endpoint, DB URL, model name,
guardrail thresholds) comes from environment variables, so the same code runs in pytest,
on a laptop (SQLite + local files), in docker-compose (Postgres + Azurite) and against
real Azure without code changes (12-factor style). Settings are read on each call to
``get_settings()`` rather than cached, which keeps tests that monkeypatch env vars simple.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load a local .env when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _float_or_none(name: str) -> float | None:
    raw = _env(name)
    return float(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    storage_backend: str  # local | azurite | azure
    azure_account: str
    azure_key: str
    azure_scheme: str  # wasbs | abfss (only used when storage_backend == "azure")
    azure_connection_string: str
    azure_client_id: str
    azure_client_secret: str
    azure_tenant_id: str
    raw_container: str
    curated_container: str
    spark_master: str
    spark_jars_packages: str
    openai_model: str
    triage_confidence_threshold: float
    triage_max_input_tokens: int
    triage_max_output_tokens: int
    triage_tokenizer: str
    masking_salt: str
    price_input_per_1m: float | None
    price_output_per_1m: float | None
    triage_enabled: bool

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def curated_dir(self) -> Path:
        return self.data_dir / "curated"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"


def get_settings() -> Settings:
    data_dir = Path(_env("DATA_DIR", str(PROJECT_ROOT / "data")))
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    default_db = f"sqlite:///{(data_dir / 'local.db').as_posix()}"
    return Settings(
        data_dir=data_dir,
        database_url=_env("DATABASE_URL", default_db),
        storage_backend=_env("STORAGE_BACKEND", "local").lower(),
        azure_account=_env("AZURE_STORAGE_ACCOUNT", "devstoreaccount1"),
        azure_key=_env("AZURE_STORAGE_KEY"),
        azure_scheme=_env("AZURE_STORAGE_SCHEME", "abfss").lower(),
        azure_connection_string=_env("AZURE_STORAGE_CONNECTION_STRING"),
        azure_client_id=_env("AZURE_CLIENT_ID"),
        azure_client_secret=_env("AZURE_CLIENT_SECRET"),
        azure_tenant_id=_env("AZURE_TENANT_ID"),
        raw_container=_env("RAW_CONTAINER", "raw"),
        curated_container=_env("CURATED_CONTAINER", "curated"),
        spark_master=_env("SPARK_MASTER_URL", "local[*]"),
        spark_jars_packages=_env(
            "SPARK_JARS_PACKAGES",
            "org.apache.hadoop:hadoop-azure:3.3.4,com.microsoft.azure:azure-storage:8.6.6,"
            "org.postgresql:postgresql:42.7.3",
        ),
        openai_model=_env("OPENAI_MODEL", "gpt-4o-mini"),
        triage_confidence_threshold=float(_env("TRIAGE_CONFIDENCE_THRESHOLD", "0.70")),
        triage_max_input_tokens=int(_env("TRIAGE_MAX_INPUT_TOKENS", "3000")),
        triage_max_output_tokens=int(_env("TRIAGE_MAX_OUTPUT_TOKENS", "400")),
        triage_tokenizer=_env("TRIAGE_TOKENIZER", "tiktoken").lower(),
        masking_salt=_env("MASKING_SALT", "change-me"),
        price_input_per_1m=_float_or_none("OPENAI_PRICE_INPUT_PER_1M"),
        price_output_per_1m=_float_or_none("OPENAI_PRICE_OUTPUT_PER_1M"),
        triage_enabled=_env("TRIAGE_ENABLED", "false").lower() in {"1", "true", "yes"},
    )
