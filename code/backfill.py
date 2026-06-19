"""Backfill safe-default (NEI) rows in output.csv with real predictions.

Re-processes ONLY the rows whose claim_status is 'not_enough_information' AND
issue_type is 'unknown' (the safe-default signature), leaving all real
predictions untouched. Patches results in place. Safe to re-run; skips rows
that are already real or that still fail after the run.

Usage (from code/):
    python backfill.py                 # backfill all NEI rows in output.csv
    python backfill.py --strategy single
    python backfill.py --dry-run       # show which rows would be backfilled
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import agent  # noqa: E402
from config import OUTPUT_CSV, OUTPUT_COLUMNS  # noqa: E402
from data_loader import (  # noqa: E402
    load_all_data, get_user_history, compute_history_risk,
    get_evidence_requirement,
)


def is_safe_default(row) -> bool:
    """A safe-default row: NEI + unknown issue + unknown part + unknown sev."""
    return (str(row.get("claim_status", "")).strip().lower() == "not_enough_information"
            and str(row.get("issue_type", "")).strip().lower() == "unknown"
            and str(row.get("severity", "")).strip().lower() == "unknown")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Backfill safe-default rows in output.csv.")
    p.add_argument("--strategy", default="two", choices=["two", "single"])
    p.add_argument("--dry-run", action="store_true", help="show rows, make no calls")
    p.add_argument("--output", default=str(OUTPUT_CSV))
    args = p.parse_args(argv)

    df = pd.read_csv(args.output)
    mask = df.apply(is_safe_default, axis=1)
    targets = df[mask]
    print(f"output.csv: {len(df)} rows; {len(targets)} safe-default rows to backfill.")
    if args.dry_run:
        for _, r in targets.iterrows():
            print(f"  would backfill {r['user_id']} ({r['claim_object']})")
        return 0
    if len(targets) == 0:
        print("Nothing to backfill — all rows are real predictions.")
        return 0

    # Build the key pool + load lookup data.
    try:
        from keypool import build_pool
        pool = build_pool()
        agent.set_key_pool(pool)
        print(f"Key pool: {len(pool)} keys, {pool.live_count()} live.")
        if pool.live_count() == 0:
            print("No live keys — cannot backfill now. Wait for quota reset.")
            return 1
    except Exception as err:
        print(f"No key pool ({err}); cannot backfill.")
        return 1

    _, history, evidence = load_all_data(use_sample=False)
    fixed = 0
    for idx, row in targets.iterrows():
        hist = get_user_history(history, row["user_id"])
        history_risk = compute_history_risk(hist)
        ev = get_evidence_requirement(evidence, row["claim_object"])
        try:
            out = agent.analyze_claim_by_strategy(
                client=None, strategy=args.strategy,
                user_claim=row["user_claim"], claim_object=row["claim_object"],
                image_paths=row["image_paths"], evidence_requirement=ev,
                history_risk=history_risk,
            )
            if str(out["claim_status"]).lower() == "not_enough_information" \
               and str(out["issue_type"]).lower() == "unknown":
                print(f"  {row['user_id']}: still failed (quota?) — skipped.")
                continue
            rec = {
                "user_id": row["user_id"], "image_paths": row["image_paths"],
                "user_claim": row["user_claim"], "claim_object": row["claim_object"],
            }
            rec.update(out)
            rec["valid_image"] = "true" if out["valid_image"] else "false"
            rec["evidence_standard_met"] = "true" if out["evidence_standard_met"] else "false"
            for c in OUTPUT_COLUMNS:
                df.at[idx, c] = rec.get(c, "")
            fixed += 1
            print(f"  {row['user_id']}: BACKFILLED -> {out['claim_status']} | "
                  f"{out['issue_type']} | {out['object_part']} | {out['severity']}")
        except Exception as err:
            print(f"  {row['user_id']}: error ({err}) — skipped.")

    df.to_csv(args.output, index=False)
    real = (df["claim_status"] != "not_enough_information").sum()
    print(f"\nBackfilled {fixed} rows. output.csv now has {real}/{len(df)} real predictions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
