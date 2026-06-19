"""Deterministic postprocessor.

The sole place that guarantees output conforms to specs/data_contract.md and
the (data-grounded) consistency invariants in specs/behavior_spec.md. The
model is never trusted to emit contract-conforming values; everything it
returns passes through here first.

Invariants enforced (data-grounded; see behavior_spec.md scenario 7 refined):
  * claim_status=supported  =>  valid_image=True AND evidence_standard_met=True
                                 AND supporting_image_ids has >=1 valid id
  * evidence_standard_met=False  =>  claim_status != supported
  * valid_image=False            =>  claim_status != supported
  (NOTE: valid_image=False does NOT force evidence_standard_met=False.)
"""
import difflib
from typing import Any, Dict, List

from config import (
    VALID_CLAIM_STATUS, VALID_ISSUE_TYPES, VALID_SEVERITY, VALID_RISK_FLAGS,
    PARTS_MAP,
)
from image_utils import parse_image_paths, image_id_from_path


# Fallback sentinels for unfixable values.
_FALLBACK = {
    "claim_status": "not_enough_information",
    "issue_type": "unknown",
    "object_part": "unknown",
    "severity": "unknown",
}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _fuzzy_fix(value: Any, allowed: List[str], fallback: str) -> str:
    """Lowercase+strip, then exact or difflib closest-match against allowed.

    Cutoff 0.6: if nothing is close enough, return ``fallback``.
    """
    v = _norm(value)
    if v in allowed:
        return v
    if v:
        matches = difflib.get_close_matches(v, allowed, n=1, cutoff=0.6)
        if matches:
            return matches[0]
    return fallback


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    n = _norm(value)
    if n in ("true", "1", "yes"):
        return True
    if n in ("false", "0", "no"):
        return False
    return default


def _fill_str(value: Any, sentinel: str) -> str:
    if value is None:
        return sentinel
    s = str(value).strip()
    return s if s else sentinel


def _clean_risk_flags(raw: Any, history_risk: bool) -> str:
    """Split, normalize, validate, inject history flag, rejoin; 'none' if empty."""
    if not raw:
        tokens: List[str] = []
    else:
        tokens = [_norm(t) for t in str(raw).split(";")]
    valid = [t for t in tokens if t in VALID_RISK_FLAGS and t != "none"]
    # de-duplicate, preserve order
    seen = set()
    uniq = []
    for t in valid:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if history_risk and "user_history_risk" not in seen:
        uniq.append("user_history_risk")
    return ";".join(uniq) if uniq else "none"


def _clean_supporting_ids(raw: Any, submitted_ids: List[str]) -> str:
    """Keep only IDs actually present in the row's image_paths; 'none' if empty."""
    submitted = set(submitted_ids)
    if not raw:
        return "none"
    tokens = [t.strip() for t in str(raw).split(";") if t and t.strip() != "none"]
    kept = [t for t in tokens if t in submitted]
    # de-duplicate, preserve order
    seen = set()
    uniq = []
    for t in kept:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return ";".join(uniq) if uniq else "none"


def postprocess(analysis: Dict[str, Any], claim_object: str,
                image_paths: str, history_risk: bool) -> Dict[str, Any]:
    """Validate and fix a model analysis dict into a contract-conforming dict.

    Args:
        analysis: raw model output (keys matching ClaimAnalysis fields).
        claim_object: 'car' | 'laptop' | 'package'.
        image_paths: the row's raw image_paths cell (for ID filtering).
        history_risk: rule-based history risk signal (drives flag injection).
    """
    a = analysis or {}

    # Parse the submitted image IDs once.
    submitted_ids = [image_id_from_path(p) for p in parse_image_paths(image_paths)]

    allowed_parts = PARTS_MAP.get(claim_object, PARTS_MAP["car"])  # unknown object -> any, fallback safe

    # --- Rule 1 & 2: enum + object-bound part ---
    issue_type = _fuzzy_fix(a.get("issue_type"), VALID_ISSUE_TYPES, _FALLBACK["issue_type"])
    severity = _fuzzy_fix(a.get("severity"), VALID_SEVERITY, _FALLBACK["severity"])
    claim_status = _fuzzy_fix(a.get("claim_status"), VALID_CLAIM_STATUS, _FALLBACK["claim_status"])
    object_part = _fuzzy_fix(a.get("object_part"), allowed_parts, _FALLBACK["object_part"])

    # --- Rule 3 & 4: risk flags + history injection ---
    risk_flags = _clean_risk_flags(a.get("risk_flags"), history_risk)

    # --- Rule 5/6: booleans + supporting IDs ---
    valid_image = _coerce_bool(a.get("valid_image"), True)
    evidence_standard_met = _coerce_bool(a.get("evidence_standard_met"), True)
    supporting_image_ids = _clean_supporting_ids(a.get("supporting_image_ids"), submitted_ids)
    has_supporting = supporting_image_ids != "none"

    # --- Rule 7: no nulls ---
    evidence_standard_met_reason = _fill_str(a.get("evidence_standard_met_reason"),
                                             "Evidence standard not evaluated.")
    claim_status_justification = _fill_str(a.get("claim_status_justification"),
                                           "No justification provided.")

    # --- Invariants: supported requires valid_image AND evidence_met AND a supporting id ---
    if claim_status == "supported":
        if not valid_image or not evidence_standard_met or not has_supporting:
            claim_status = "not_enough_information"
    # evidence_standard_met False => not supported
    if not evidence_standard_met and claim_status == "supported":
        claim_status = "not_enough_information"
    # valid_image False => not supported
    if not valid_image and claim_status == "supported":
        claim_status = "not_enough_information"

    return {
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": evidence_standard_met_reason,
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": claim_status_justification,
        "supporting_image_ids": supporting_image_ids,
        "valid_image": valid_image,
        "severity": severity,
    }
