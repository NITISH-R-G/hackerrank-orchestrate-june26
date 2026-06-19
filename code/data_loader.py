"""Data loading and rule-based risk derivation.

Implements the data-layer contract from specs/data_contract.md and the
history-risk rules referenced by specs/behavior_spec.md scenarios 6 & 10.

History risk is RULE-BASED ONLY: never delegated to the model (see
specs/agent_contract.md §7).
"""
import pandas as pd

from config import (
    CLAIMS_CSV, SAMPLE_CLAIMS_CSV, USER_HISTORY_CSV, EVIDENCE_CSV,
)


def load_all_data(use_sample: bool = False):
    """Load claims, user_history and evidence_requirements as DataFrames.

    Args:
        use_sample: if True read sample_claims.csv, else claims.csv.
    """
    claims_path = SAMPLE_CLAIMS_CSV if use_sample else CLAIMS_CSV
    claims = pd.read_csv(claims_path)
    history = pd.read_csv(USER_HISTORY_CSV)
    evidence = pd.read_csv(EVIDENCE_CSV)
    return claims, history, evidence


def get_user_history(history_df: pd.DataFrame, user_id: str) -> dict:
    """Return the history row for ``user_id`` as a dict, or {} if absent.

    Scenario 10: a missing user yields an empty dict (no crash, no risk).
    """
    row = history_df[history_df["user_id"] == user_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# Risk keywords that, if present in flags or summary, mark a user as risky.
_RISK_WORDS = ["fraud", "suspicious", "review", "flag", "reject", "deny"]


def compute_history_risk(history_row: dict) -> bool:
    """Rule-based history-risk signal. Returns True if any rule triggers.

    Rules (any one triggers):
      * rejected / past_claim_count > 0.3   (reject ratio high)
      * last_90_days_claim_count >= 3       (recent volume high)
      * a risk keyword appears in history_flags or history_summary

    Robust to missing keys, zero totals, and bad types. Returns False on any
    unexpected failure rather than crashing the pipeline.
    """
    if not history_row:
        return False
    try:
        total = int(history_row.get("past_claim_count", 0) or 0)
        rejected = int(history_row.get("rejected_claim", 0) or 0)
        last_90 = int(history_row.get("last_90_days_claim_count", 0) or 0)
        flags = str(history_row.get("history_flags", "") or "").lower()
        summary = str(history_row.get("history_summary", "") or "").lower()

        if total > 0 and (rejected / total) > 0.3:
            return True
        if last_90 >= 3:
            return True
        if any(word in flags for word in _RISK_WORDS):
            return True
        if any(word in summary for word in _RISK_WORDS):
            return True
    except Exception:
        return False
    return False


def get_evidence_requirement(
    evidence_df: pd.DataFrame, claim_object: str, issue_family: str = None
) -> str:
    """Return the minimum_image_evidence text for the matched requirement.

    Matching precedence:
      1. exact (case-insensitive) match on applies_to for this claim_object
         or 'all';
      2. substring match (issue_family tokens appear in applies_to);
      3. fallback to the 'reviewability' general requirement;
      4. final fallback: the first row's text.

    Never raises; always returns a non-empty string.
    """
    df = evidence_df.copy()
    applies_to_l = df["applies_to"].astype(str).str.lower().to_numpy()
    obj_mask = df["claim_object"].isin([claim_object, "all"]).to_numpy()

    family = (issue_family or "").strip().lower()

    # 1. exact applies_to match
    if family and family != "unknown":
        exact_mask = obj_mask & (applies_to_l == family)
        if exact_mask.any():
            return str(df.loc[exact_mask, "minimum_image_evidence"].iloc[0])

        # 2. substring / token overlap match
        family_tokens = {t.strip() for t in family.replace(",", " ").split() if t.strip()}
        for i in range(len(df)):
            if not obj_mask[i]:
                continue
            row_tokens = {t.strip() for t in applies_to_l[i].replace(",", " ").split() if t.strip()}
            if family_tokens and family_tokens & row_tokens:
                return str(df.iloc[i]["minimum_image_evidence"])

    # 3. reviewability fallback
    review_mask = applies_to_l == "reviewability"
    if review_mask.any():
        return str(df.loc[review_mask, "minimum_image_evidence"].iloc[0])

    # 4. final fallback
    return str(df.iloc[0]["minimum_image_evidence"])
