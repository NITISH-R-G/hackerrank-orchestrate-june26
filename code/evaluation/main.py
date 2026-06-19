"""Evaluation harness: run both strategies on sample_claims.csv and report.

Specs: specs/evaluation_spec.md. Runs Strategy A (single-pass) and Strategy B
(two-pass) on all 20 labeled rows, scores them with evaluation/metrics.py,
auto-selects the winner by mean accuracy (tie -> two-pass), and writes
summary.json + per-strategy prediction CSVs.

Usage (run from the code/ directory):

    # dry-run on the first N rows (cheap wiring check)
    python -m evaluation.main --limit 2

    # full evaluation on all 20 samples (live Gemini calls)
    python -m evaluation.main --full

    # re-score from cached predictions without new API calls
    python -m evaluation.main --rescore
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Make code/ importable when run as a module or script.
HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import agent  # noqa: E402
from config import (  # noqa: E402
    SAMPLE_CLAIMS_CSV, USER_HISTORY_CSV, EVIDENCE_CSV, OUTPUT_COLUMNS,
)
from data_loader import (  # noqa: E402
    load_all_data, get_user_history, compute_history_risk,
    get_evidence_requirement,
)
from evaluation.metrics import summarize  # noqa: E402

EVAL_DIR = HERE
PRED_DIR = EVAL_DIR / "predictions"
PRED_DIR.mkdir(exist_ok=True)
SUMMARY_JSON = EVAL_DIR / "summary.json"


def _gt_dicts(df: pd.DataFrame) -> list:
    """Ground-truth rows as dicts over the scored fields."""
    return [df.iloc[i].to_dict() for i in range(len(df))]


def _run_strategy(strategy: str, df: pd.DataFrame, history_df, evidence_df,
                  limit=None) -> list:
    """Run one strategy over the sample rows; return (preds, stats, elapsed)."""
    client = agent.build_client()
    # Install a key pool for rotation across the 20/day-per-key free-tier quota.
    try:
        from keypool import build_pool
        agent.set_key_pool(build_pool())
        pool_size = len(build_pool())
        print(f"  key pool size: {pool_size}")
    except Exception as e:
        print(f"  (single-key mode: {e})")
    agent.reset_stats()
    n = len(df) if limit is None else min(limit, len(df))
    preds = []
    t0 = time.time()
    for i in range(n):
        row = df.iloc[i]
        user_id = row["user_id"]
        hist = get_user_history(history_df, user_id)
        history_risk = compute_history_risk(hist)
        ev = get_evidence_requirement(evidence_df, row["claim_object"])
        try:
            out = agent.analyze_claim_by_strategy(
                client, strategy=strategy,
                user_claim=row["user_claim"],
                claim_object=row["claim_object"],
                image_paths=row["image_paths"],
                evidence_requirement=ev, history_risk=history_risk,
            )
        except Exception as err:  # per-row isolation: never abort the batch
            print(f"  [{strategy}] row {i} ({user_id}) failed: {err}")
            out = agent.safe_default_analysis(history_risk)
        rec = {
            "user_id": user_id, "image_paths": row["image_paths"],
            "user_claim": row["user_claim"], "claim_object": row["claim_object"],
        }
        rec.update(out)
        rec["valid_image"] = "true" if out["valid_image"] else "false"
        rec["evidence_standard_met"] = "true" if out["evidence_standard_met"] else "false"
        preds.append(rec)
        print(f"  [{strategy}] {i+1}/{n} {user_id}: {out['claim_status']} "
              f"| {out['issue_type']} | {out['object_part']} | {out['severity']}")
    elapsed = time.time() - t0
    return preds, dict(agent.stats), elapsed


def _write_pred_csv(preds: list, path: Path) -> None:
    rows = [{c: p.get(c, "") for c in OUTPUT_COLUMNS} for p in preds]
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(path, index=False)


def _pred_dicts_for_scoring(preds: list) -> list:
    out = []
    for p in preds:
        out.append({
            "claim_status": p["claim_status"], "issue_type": p["issue_type"],
            "object_part": p["object_part"], "severity": p["severity"],
            "valid_image": str(p["valid_image"]).lower(),
            "evidence_standard_met": str(p["evidence_standard_met"]).lower(),
        })
    return out


def run_evaluation(limit=None, strategies=("single", "two")) -> dict:
    claims, history, evidence = load_all_data(use_sample=True)
    gt = _gt_dicts(claims)

    results = {}
    for strat in strategies:
        print(f"\n=== Strategy {strat} ===")
        preds, stats, elapsed = _run_strategy(
            strat, claims, history, evidence, limit=limit)
        _write_pred_csv(preds, PRED_DIR / f"sample_pred_{strat}.csv")
        summary = summarize(gt[:len(preds)], _pred_dicts_for_scoring(preds))
        summary["stats"] = stats
        summary["runtime_seconds"] = round(elapsed, 2)
        summary["avg_latency_seconds"] = round(elapsed / len(preds), 2) if preds else 0.0
        results[strat] = summary
        print(f"  mean_accuracy={summary['mean_accuracy']:.3f}  "
              f"overall_exact={summary['overall_exact_match']:.3f}  "
              f"calls={stats}  runtime={elapsed:.1f}s")

    # Auto-select winner: higher mean accuracy; tie -> two-pass.
    if len(results) == 2:
        a, b = strategies
        if results[a]["mean_accuracy"] > results[b]["mean_accuracy"]:
            winner = a
        elif results[b]["mean_accuracy"] > results[a]["mean_accuracy"]:
            winner = b
        else:
            winner = "two" if "two" in strategies else a
    else:
        winner = max(results, key=lambda k: results[k]["mean_accuracy"])
    results["winner"] = winner
    results["limit"] = limit
    print(f"\n=== WINNER: {winner} ===")
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate both strategies on samples.")
    p.add_argument("--full", action="store_true", help="run all 20 rows (live)")
    p.add_argument("--limit", type=int, default=None, help="dry-run first N rows")
    p.add_argument("--rescore", action="store_true",
                   help="re-score from cached prediction CSVs (no API calls)")
    args = p.parse_args(argv)

    if args.rescore:
        claims, _, _ = load_all_data(use_sample=True)
        gt = _gt_dicts(claims)
        results = {}
        for strat in ("single", "two"):
            path = PRED_DIR / f"sample_pred_{strat}.csv"
            if not path.exists():
                print(f"No cached predictions at {path}; run --full first.")
                return 1
            df = pd.read_csv(path)
            results[strat] = summarize(gt, _pred_dicts_for_scoring(df.to_dict("records")))
        results["winner"] = max(
            ("single", "two"),
            key=lambda k: results[k]["mean_accuracy"],
        )
    else:
        limit = None if args.full else (args.limit if args.limit else 2)
        results = run_evaluation(limit=limit)

    SUMMARY_JSON.write_text(json.dumps(results, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nWrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
