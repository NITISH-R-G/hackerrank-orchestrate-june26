"""Main entry point for the claim verification pipeline.

Runs the two-pass (or single-pass) VLM agent over `dataset/claims.csv` and
writes `output.csv` (one row per input claim, exact column order per the data
contract). Designed to be crash-safe and resumable:

  * Each completed row is written immediately (append-after-header), so a
    crash mid-run keeps already-finished predictions.
  * On resume, cached rows (agent on-disk cache) are reused without new calls.
  * If the model/quota fails for a row, that row is filled with a contract-
    compliant safe default and the run continues — output.csv is always
    complete and valid.

Usage (from code/):
    python main.py                         # default: two-pass on claims.csv
    python main.py --strategy single       # single-pass
    python main.py --limit 10              # first 10 rows only
    python main.py --input X.csv --output Y.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Adjust imports according to actual available modules (like config)
try:
    from config import (CLAIMS_CSV, OUTPUT_CSV, USER_HISTORY_CSV, EVIDENCE_CSV,
                        OUTPUT_COLUMNS)
except ImportError:
    # Fallback paths/values for testing if config is not available.
    CLAIMS_CSV = "dataset/claims.csv"
    OUTPUT_CSV = "output.csv"
    USER_HISTORY_CSV = "dataset/user_history.csv"
    EVIDENCE_CSV = "dataset/evidence_requirements.csv"
    OUTPUT_COLUMNS = [
        "user_id", "image_paths", "user_claim", "claim_object",
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part", "claim_status",
        "claim_status_justification", "supporting_image_ids",
        "valid_image", "severity",
    ]


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _safe_default_row(row, history_risk: bool) -> dict:
    """Build a contract-compliant output row from a safe-default analysis."""
    from agent import safe_default_analysis
    d = safe_default_analysis(history_risk)
    out = {
        "user_id": row["user_id"], "image_paths": row["image_paths"],
        "user_claim": row["user_claim"], "claim_object": row["claim_object"],
    }
    out.update(d)
    out["valid_image"] = "true" if d["valid_image"] else "false"
    out["evidence_standard_met"] = "true" if d["evidence_standard_met"] else "false"
    return out


def run_pipeline(claims_path: str, output_path: str, *, strategy: str = "two",
                 limit: int = None, _runner=None) -> int:
    """Execute the main processing pipeline; return rows written.

    Args:
        claims_path: input claims CSV.
        output_path: where to write output.csv.
        strategy: 'two' (two-pass) or 'single' (single-pass).
        limit: optional cap on rows processed.
        _runner: injectable analyze fn (row, history_risk, evidence_req) -> dict,
                 for testing. Defaults to the live agent.
    """
    logging.info(f"Starting pipeline. Input: {claims_path}, Output: {output_path}")
    claims = pd.read_csv(claims_path)
    history = pd.read_csv(USER_HISTORY_CSV)
    evidence = pd.read_csv(EVIDENCE_CSV)

    if _runner is None:
        import agent
        from data_loader import (get_user_history, compute_history_risk,
                                 get_evidence_requirement)
        from keypool import build_pool
        # Install the key pool so daily-quota 429s rotate across all keys.
        try:
            pool = build_pool()
            agent.set_key_pool(pool)
            logger.info("Key pool installed: %d keys (%d live).",
                        len(pool), pool.live_count())
        except Exception as err:
            logger.warning("Could not build key pool (%s); single-key mode.", err)

        def _live_runner(row, history_risk, evidence_req):
            return agent.analyze_claim_by_strategy(
                client=None, strategy=strategy,
                user_claim=row["user_claim"], claim_object=row["claim_object"],
                image_paths=row["image_paths"], evidence_requirement=evidence_req,
                history_risk=history_risk,
            )

        def _history_lookup(uid):
            return get_user_history(history, uid)

        _runner = _live_runner
        _hist_risk = lambda uid: compute_history_risk(_history_lookup(uid))  # noqa: E731
        _ev_req = lambda obj: get_evidence_requirement(evidence, obj)  # noqa: E731
    else:
        _hist_risk = lambda uid: False  # noqa: E731
        _ev_req = lambda obj: "general"  # noqa: E731

    n = len(claims) if limit is None else min(limit, len(claims))

    # Write header; incrementally append rows (crash-safe).
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(f, index=False)

    written = 0
    for i in range(n):
        row = claims.iloc[i]
        history_risk = _hist_risk(row["user_id"])
        evidence_req = _ev_req(row["claim_object"])
        try:
            out = _runner(row, history_risk, evidence_req)
            rec = {
                "user_id": row["user_id"], "image_paths": row["image_paths"],
                "user_claim": row["user_claim"], "claim_object": row["claim_object"],
            }
            rec.update(out)
            rec["valid_image"] = "true" if out["valid_image"] else "false"
            rec["evidence_standard_met"] = "true" if out["evidence_standard_met"] else "false"
        except Exception as err:
            logger.warning("row %d (%s) failed (%s); safe-defaulting.",
                           i, row.get("user_id", "?"), err)
            rec = _safe_default_row(row, history_risk)
        with open(out_path, "a", encoding="utf-8", newline="") as f:
            pd.DataFrame([rec], columns=OUTPUT_COLUMNS).to_csv(
                f, index=False, header=False)
        written += 1
        logger.info("[%d/%d] %s: %s | %s | %s | %s", i + 1, n, row["user_id"],
                    rec.get("claim_status"), rec.get("issue_type"),
                    rec.get("object_part"), rec.get("severity"))
    logging.info(f"Pipeline execution completed. {written} rows -> {output_path}")
    return written


def main(args: list = None) -> int:
    """Parse arguments and start the pipeline. Returns 0 on success."""
    parser = argparse.ArgumentParser(
        description="Verify damage claims using multi-modal evidence."
    )
    parser.add_argument("--input", type=str, default=str(CLAIMS_CSV),
                        help="Path to input claims CSV file")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV),
                        help="Path to write output predictions CSV")
    parser.add_argument("--user-history", type=str, default=str(USER_HISTORY_CSV),
                        help="Path to user history CSV file")
    parser.add_argument("--evidence-reqs", type=str, default=str(EVIDENCE_CSV),
                        help="Path to evidence requirements CSV file")
    parser.add_argument("--strategy", type=str, default="two", choices=["two", "single"],
                        help="Analysis strategy: 'two' (two-pass) or 'single'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows")

    try:
        parsed_args = parser.parse_args(args)
        # Only pass strategy/limit when non-default, so the minimal call shape
        # run_pipeline(claims_path=..., output_path=...) is preserved.
        extra = {}
        if parsed_args.strategy != "two":
            extra["strategy"] = parsed_args.strategy
        if parsed_args.limit is not None:
            extra["limit"] = parsed_args.limit
        run_pipeline(
            claims_path=parsed_args.input,
            output_path=parsed_args.output,
            **extra,
        )
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
