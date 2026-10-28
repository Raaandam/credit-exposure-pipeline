"""Airflow DAG: daily, idempotent, incremental credit exposure load.

    wait_for_manifest -> spark_exposure_job -> spark_dq_job -> llm_triage_suggestions (optional)

Design choices:
* One DAG run == one business date (``{{ ds }}``). Every task reads/writes only that
  partition, so retries, clears and backfills (``catchup=True``) are safe and do not touch
  other days. ``max_active_runs=1`` keeps backfills ordered, which matters because the
  triage context references the previous successful load.
* The schedule fires 06:00 Mon-Fri; Airflow's ``ds`` is the start of the data interval, i.e.
  the previous business day's close - the date the source extracts are for.
* The sensor waits for ``manifest.json`` (written last by the source) in reschedule mode,
  so it does not occupy a worker slot while waiting.
* Spark is invoked with ``spark-submit`` through BashOperator: explicit, easy to reproduce
  by hand, and no provider-version coupling. SPARK_MASTER_URL decides local[*] vs cluster.
* DQ runs with ``trigger_rule=all_done``: if the exposure job fails because a feed is
  missing, DQ still records the break - the dashboard must show the gap.
* LLM triage only *suggests*; nothing is applied without human approval in the dashboard.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sensors.python import PythonSensor

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/project")
SPARK_SUBMIT = (
    "spark-submit --master ${SPARK_MASTER_URL:-local[*]} --packages ${SPARK_JARS_PACKAGES} "
    "--conf spark.driver.memory=2g "
)


def _manifest_landed(business_date: str) -> bool:
    from exposure_pipeline.storage.blob import manifest_exists

    return manifest_exists(business_date)


def _triage_enabled() -> bool:
    return os.getenv("TRIAGE_ENABLED", "false").lower() in {"1", "true", "yes"} and bool(os.getenv("OPENAI_API_KEY"))


def _run_triage(business_date: str) -> int:
    from datetime import date

    from exposure_pipeline.triage.runner import run_triage

    return run_triage(business_date=date.fromisoformat(business_date))


default_args = {
    "owner": "credit-risk-data",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
}

with DAG(
    dag_id="credit_exposure_daily",
    description="Counterparty/facility/collateral/rating feeds -> curated exposure + DQ results",
    schedule="0 6 * * 1-5",
    start_date=datetime(2026, 1, 5),
    catchup=True,
    max_active_runs=1,
    default_args=default_args,
    tags=["credit-risk", "exposure", "data-quality"],
) as dag:
    wait_for_manifest = PythonSensor(
        task_id="wait_for_manifest",
        python_callable=_manifest_landed,
        op_kwargs={"business_date": "{{ ds }}"},
        mode="reschedule",
        poke_interval=300,
        timeout=6 * 60 * 60,
    )

    spark_exposure_job = BashOperator(
        task_id="spark_exposure_job",
        bash_command=SPARK_SUBMIT + f"{PROJECT_DIR}/src/exposure_pipeline/spark_jobs/daily_exposure_job.py --business-date {{{{ ds }}}}",
        execution_timeout=timedelta(hours=1),
    )

    spark_dq_job = BashOperator(
        task_id="spark_dq_job",
        bash_command=SPARK_SUBMIT + f"{PROJECT_DIR}/src/exposure_pipeline/spark_jobs/dq_job.py --business-date {{{{ ds }}}}",
        trigger_rule="all_done",
        execution_timeout=timedelta(minutes=30),
    )

    triage_enabled = ShortCircuitOperator(
        task_id="triage_enabled",
        python_callable=_triage_enabled,
        ignore_downstream_trigger_rules=True,
    )

    llm_triage_suggestions = PythonOperator(
        task_id="llm_triage_suggestions",
        python_callable=_run_triage,
        op_kwargs={"business_date": "{{ ds }}"},
        retries=0,  # LLM failures fall back to manual review inside the task; don't re-bill on retry
    )

    wait_for_manifest >> spark_exposure_job >> spark_dq_job >> triage_enabled >> llm_triage_suggestions
