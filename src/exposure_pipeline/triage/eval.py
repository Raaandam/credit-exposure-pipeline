"""Evaluate LLM triage against ground truth.

Two label sources, reported separately (they answer different questions):

* **Generator labels** (``data/labels/error_labels.csv``): the injected root cause. A break's
  true category is the majority label among its failing keys (record-level checks) or the
  feed-level incident for that day/feed (aggregate checks). Breaks with no matching label are
  reported as "unlabelled" and excluded, not guessed.
* **Analyst labels**: ``eval/hand_labels.csv`` (business_date, rule_id, analyst_category) for
  ~150 breaks labelled by hand, optionally plus the categories reviewers chose in the
  dashboard (``--include-reviewer-labels``). This is the "agreement with analyst labels"
  number; hand labels win where both exist.

Metrics: agreement % (fallbacks to needs_manual_review count as disagreement), coverage
(share auto-triaged), agreement on auto-triaged breaks, raw model agreement before the
confidence gate, per-class precision/recall, confusion matrix, confidence calibration,
latency and token/cost per break.

    python -m exposure_pipeline.triage.eval --update-readme
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from exposure_pipeline.config import PROJECT_ROOT, get_settings
from exposure_pipeline.reporting import update_readme_section
from exposure_pipeline.storage.db import get_engine, read_sql
from exposure_pipeline.triage.schemas import ALL_CATEGORIES, NEEDS_MANUAL_REVIEW

HAND_LABELS = PROJECT_ROOT / "eval" / "hand_labels.csv"


# ----------------------------------------------------------------------------- ground truth
def derive_ground_truth(breaks: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    """Map each break (result_id) to its injected root-cause category, or None."""
    labels = labels.copy()
    labels["business_date"] = labels["business_date"].astype(str).str[:10]
    rec = labels[labels["level"] == "record"]
    rec_map = {(d, f, k): e for d, f, k, e in rec[["business_date", "feed", "record_key", "error_type"]].itertuples(index=False)}
    feed_map = {(d, f): e for d, f, e in labels[labels["level"] == "feed"][["business_date", "feed", "error_type"]].itertuples(index=False)}

    truth = {}
    for row in breaks.itertuples(index=False):
        bd = str(row.business_date)[:10]
        keys = row.failed_keys if isinstance(row.failed_keys, list) else json.loads(row.failed_keys or "[]")
        cats = [rec_map[(bd, row.feed, str(k))] for k in keys if (bd, row.feed, str(k)) in rec_map]
        if cats:
            truth[row.result_id] = Counter(cats).most_common(1)[0][0]
        else:
            truth[row.result_id] = feed_map.get((bd, row.feed))
    return pd.Series(truth, name="truth", dtype=object)


def load_hand_labels(path: Path = HAND_LABELS) -> pd.Series:
    if not Path(path).exists():
        return pd.Series(dtype=object, name="analyst")
    df = pd.read_csv(path, comment="#", dtype=str).dropna(subset=["business_date", "rule_id", "analyst_category"])
    bad = set(df["analyst_category"]) - set(ALL_CATEGORIES)
    if bad:
        raise ValueError(f"unknown categories in {path}: {sorted(bad)}")
    idx = df["business_date"].str[:10] + ":" + df["rule_id"]
    return pd.Series(df["analyst_category"].to_numpy(), index=idx, name="analyst")


# ----------------------------------------------------------------------------- data access
def load_eval_frame(engine: Engine) -> pd.DataFrame:
    breaks = read_sql(engine, "SELECT result_id, business_date, rule_id, feed, failed_keys FROM dq_results WHERE status = 'FAIL'")
    audit = read_sql(engine, """
        SELECT result_id, created_at, suggested_category, llm_category, confidence, fallback_reason, latency_ms,
               input_tokens, output_tokens, cost_usd, reviewer_decision, final_category
        FROM triage_audit""")
    if not audit.empty:
        audit = audit.sort_values("created_at").groupby("result_id").tail(1)  # latest suggestion per break
    return breaks.merge(audit, on="result_id", how="left")


# ----------------------------------------------------------------------------- metrics
def compute_metrics(df: pd.DataFrame, truth_col: str) -> dict:
    d = df[df[truth_col].notna() & df["suggested_category"].notna()].copy()
    n = len(d)
    if n == 0:
        return {"n": 0}
    pred, truth = d["suggested_category"], d[truth_col]
    auto = d[pred != NEEDS_MANUAL_REVIEW]
    raw = d[d["llm_category"].notna()]

    labels = list(ALL_CATEGORIES)
    confusion = pd.crosstab(pd.Categorical(truth, categories=labels), pd.Categorical(pred, categories=labels),
                            rownames=["truth"], colnames=["predicted"], dropna=False)
    per_class = {}
    for c in labels[:-1]:
        tp = int(((pred == c) & (truth == c)).sum())
        p_den, r_den = int((pred == c).sum()), int((truth == c).sum())
        per_class[c] = {"precision": tp / p_den if p_den else None, "recall": tp / r_den if r_den else None, "support": r_den}

    buckets = pd.cut(raw["confidence"], [0, 0.5, 0.7, 0.9, 1.0], include_lowest=True)
    calib = raw.assign(ok=raw["llm_category"] == raw[truth_col]).groupby(buckets, observed=False)["ok"].agg(["mean", "count"])

    lat = d["latency_ms"].dropna()
    cost = d["cost_usd"].dropna()
    return {
        "n": n,
        "agreement": float((pred == truth).mean()),
        "coverage": float(len(auto) / n),
        "agreement_on_auto": float((auto["suggested_category"] == auto[truth_col]).mean()) if len(auto) else None,
        "raw_model_agreement": float((raw["llm_category"] == raw[truth_col]).mean()) if len(raw) else None,
        "per_class": per_class,
        "confusion": confusion,
        "calibration": calib,
        "fallback_reasons": d["fallback_reason"].dropna().str.split(":").str[0].value_counts().to_dict(),
        "latency_ms": {"mean": float(lat.mean()), "p50": float(lat.quantile(0.5)), "p95": float(lat.quantile(0.95))} if len(lat) else None,
        "tokens": {"input_mean": float(d["input_tokens"].dropna().mean()) if d["input_tokens"].notna().any() else None,
                   "output_mean": float(d["output_tokens"].dropna().mean()) if d["output_tokens"].notna().any() else None},
        "cost_usd": {"mean_per_break": float(cost.mean()), "total": float(cost.sum())} if len(cost) else None,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def metrics_markdown(title: str, m: dict) -> str:
    if not m or m.get("n", 0) == 0:
        return f"**{title}:** no labelled breaks with triage suggestions yet."
    lines = [
        f"**{title}** (n = {m['n']} breaks)",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Agreement (fallbacks count as wrong) | {_pct(m['agreement'])} |",
        f"| Coverage (auto-triaged, not needs_manual_review) | {_pct(m['coverage'])} |",
        f"| Agreement on auto-triaged breaks | {_pct(m['agreement_on_auto'])} |",
        f"| Raw model agreement before confidence gate | {_pct(m['raw_model_agreement'])} |",
    ]
    if m.get("latency_ms"):
        lines.append(f"| Latency per break (mean / p50 / p95 ms) | {m['latency_ms']['mean']:.0f} / {m['latency_ms']['p50']:.0f} / {m['latency_ms']['p95']:.0f} |")
    if m.get("tokens", {}).get("input_mean") is not None:
        lines.append(f"| Tokens per break (input / output, mean) | {m['tokens']['input_mean']:.0f} / {m['tokens']['output_mean']:.0f} |")
    lines.append(f"| Cost per break (USD, mean) | {m['cost_usd']['mean_per_break']:.5f} |" if m.get("cost_usd") else "| Cost per break | set OPENAI_PRICE_* to compute |")
    lines += ["", "Per class:", "", "| Category | Precision | Recall | Support |", "|---|---|---|---|"]
    for c, v in m["per_class"].items():
        lines.append(f"| {c} | {_pct(v['precision'])} | {_pct(v['recall'])} | {v['support']} |")
    conf = m["confusion"]
    lines += ["", "Confusion matrix (rows = truth, columns = predicted):", "",
              "| truth \\ predicted | " + " | ".join(conf.columns.astype(str)) + " |",
              "|---" * (len(conf.columns) + 1) + "|"]
    for idx, row in conf.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(int(v)) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def export_labelling_sheet(engine: Engine, path: Path, n: int = 150, seed: int = 7) -> int:
    """Write a blind labelling sheet: break evidence only, NO model suggestion, so analyst
    labels are not anchored on the LLM's answer. Stratified by rule to avoid one noisy rule
    dominating the sample."""
    df = read_sql(engine, """SELECT business_date, rule_id, rule_name, feed, observed_value, message, sample_records
                             FROM dq_results WHERE status = 'FAIL'""")
    if df.empty:
        return 0
    if len(df) <= n:
        sample = df
    else:  # up to k per rule first, then top up at random to n
        per_rule = max(1, n // df["rule_id"].nunique())
        first = df.sample(frac=1, random_state=seed).groupby("rule_id").head(per_rule)
        rest = df.drop(first.index).sample(frac=1, random_state=seed)
        sample = pd.concat([first, rest]).head(n)
    sample = sample.sort_values(["business_date", "rule_id"]).assign(analyst_category="", analyst="", notes="")
    sample["sample_records"] = sample["sample_records"].str.slice(0, 500)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    return len(sample)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description="Evaluate LLM triage vs generator and analyst labels")
    ap.add_argument("--export-sheet", type=Path, help="write a blind labelling sheet to this CSV and exit")
    ap.add_argument("--labels", type=Path, default=settings.labels_dir / "error_labels.csv")
    ap.add_argument("--hand-labels", type=Path, default=HAND_LABELS)
    ap.add_argument("--include-reviewer-labels", action="store_true",
                    help="use dashboard reviewer categories as analyst labels where no hand label exists")
    ap.add_argument("--database-url", default=settings.database_url)
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args(argv)

    if args.export_sheet:
        n = export_labelling_sheet(get_engine(args.database_url), args.export_sheet)
        print(f"wrote {n} breaks to {args.export_sheet}; fill analyst_category and copy rows into {HAND_LABELS}")
        return

    df = load_eval_frame(get_engine(args.database_url)).set_index("result_id", drop=False)
    labels = pd.read_csv(args.labels, dtype=str, keep_default_na=False)
    df["truth"] = derive_ground_truth(df.reset_index(drop=True), labels)
    analyst = load_hand_labels(args.hand_labels)
    df["analyst"] = analyst.reindex(df.index)
    if args.include_reviewer_labels:
        df["analyst"] = df["analyst"].fillna(df["final_category"].where(df["reviewer_decision"].notna()))
    df["analyst"] = df["analyst"].replace({np.nan: None})

    gen = compute_metrics(df, "truth")
    ana = compute_metrics(df, "analyst")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    unlabelled = int(df["truth"].isna().sum())
    md = "\n\n".join([
        f"_Last measured {stamp} with `python -m exposure_pipeline.triage.eval`; model: "
        f"{settings.openai_model}; confidence threshold {settings.triage_confidence_threshold:.2f}. "
        f"{len(df)} FAIL breaks, {unlabelled} without a generator label (excluded)._",
        metrics_markdown("Agreement with analyst labels", ana),
        metrics_markdown("Agreement with generator ground truth", gen),
    ])
    out_dir = settings.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "triage_eval.md").write_text(md, encoding="utf-8")
    df.drop(columns=["failed_keys"]).to_csv(out_dir / "triage_eval_rows.csv", index=False)
    print(md)
    if args.update_readme:
        print("README updated" if update_readme_section("triage_eval", md) else "README marker not found")


if __name__ == "__main__":
    main()
