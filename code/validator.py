"""Consistency validator + deterministic force-fix.

Two-stage defense after the model returns:
  1. find_inconsistencies() — detects impossible combinations; the pipeline
     re-runs Pass 2 ONCE if any are found (per user Sprint 5 choice), then
  2. force_fix() — deterministically repairs any remaining inconsistencies.

Invariants enforced (data-grounded; see Sprint 3 calibration against
sample_claims.csv):
  * supported => valid_image AND evidence_standard_met AND supporting != none
                 AND issue_type != none
  * evidence_standard_met=False => not supported
  * valid_image=False => not supported
  * issue_type=none AND severity=high => severity=unknown

Configurable: STRICT_VALID_IMAGE_IMPLIES_EVIDENCE (default False).
When True, valid_image=False additionally forces evidence_standard_met=False.
Left False by default because labeled row user_008 (valid_image=False,
evidence_standard_met=True, contradicted) shows the strict rule causes an
exact-match miss.
"""
import os

STRICT_VALID_IMAGE_IMPLIES_EVIDENCE = os.getenv(
    "STRICT_VALID_IMAGE_IMPLIES_EVIDENCE", "false"
).strip().lower() in ("1", "true", "yes")


def _has_damage_visible_words(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in (
        "visible", "clearly", "shows", "seen", "matches", "confirmed"
    )) and not any(w in t for w in (
        "no damage", "not visible", "does not show", "doesn't show",
        "cannot see", "unclear", "not enough"
    ))


def find_inconsistencies(a: dict) -> list:
    """Return a list of human-readable inconsistency messages (empty if OK)."""
    inc = []
    status = str(a.get("claim_status", "")).strip().lower()
    issue = str(a.get("issue_type", "")).strip().lower()
    sev = str(a.get("severity", "")).strip().lower()
    valid = bool(a.get("valid_image", True))
    evidence = bool(a.get("evidence_standard_met", True))
    supporting = str(a.get("supporting_image_ids", "")).strip().lower()
    justification = str(a.get("claim_status_justification", ""))

    # supported preconditions
    if status == "supported":
        if not valid:
            inc.append("supported requires valid_image=true")
        if not evidence:
            inc.append("supported requires evidence_standard_met=true")
        if supporting == "none" or not supporting:
            inc.append("supported requires at least one supporting_image_id")
        if issue == "none":
            inc.append("supported requires issue_type != none (damage must be visible)")

    # not-supported consequences
    if not evidence and status == "supported":
        inc.append("evidence_standard_met=false is incompatible with supported")
    if not valid and status == "supported":
        inc.append("valid_image=false is incompatible with supported")

    # severity vs issue
    if issue == "none" and sev == "high":
        inc.append("severity=high is incompatible with issue_type=none")

    # strict optional rule (off by default)
    if STRICT_VALID_IMAGE_IMPLIES_EVIDENCE and not valid and evidence:
        inc.append("valid_image=false forces evidence_standard_met=false (strict mode)")

    return inc


def force_fix(a: dict) -> dict:
    """Deterministically repair an analysis dict so it has no inconsistencies.

    Repairs (in priority order):
      * severity=high + issue_type=none -> severity=unknown
      * any inconsistent supported row -> claim_status=not_enough_information
    """
    out = dict(a)
    issue = str(out.get("issue_type", "")).strip().lower()
    sev = str(out.get("severity", "")).strip().lower()
    if issue == "none" and sev == "high":
        out["severity"] = "unknown"

    # Data-grounded severity calibration: scratch/dent/crack are NEVER labeled
    # 'high' in the sample (only low/medium); the only 'high' is broken_part.
    # Demote model over-prediction of 'high' for these issue types.
    _NEVER_HIGH_ISSUES = {"scratch", "dent", "crack"}
    if sev == "high" and issue in _NEVER_HIGH_ISSUES:
        out["severity"] = "medium"

    status = str(out.get("claim_status", "")).strip().lower()
    valid = bool(out.get("valid_image", True))
    evidence = bool(out.get("evidence_standard_met", True))
    supporting = str(out.get("supporting_image_ids", "")).strip().lower()
    issue = str(out.get("issue_type", "")).strip().lower()

    if status == "supported":
        broken = (
            (not valid) or (not evidence)
            or (supporting == "none" or not supporting)
            or (issue == "none")
        )
        if STRICT_VALID_IMAGE_IMPLIES_EVIDENCE and not valid and evidence:
            out["evidence_standard_met"] = False
            broken = True
        if broken:
            out["claim_status"] = "not_enough_information"
    return out
