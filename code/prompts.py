"""Prompt builders for the two-pass VLM agent (specs/agent_contract.md §1).

Pure functions — no network, no secrets — so they are unit-testable in
isolation. The agent (agent.py) composes these into model calls.

Pass 1 (text-only): extract the claimed damage family from the conversation.
Pass 2 (multimodal): analyze images against the targeted evidence requirement.
"""
from typing import List

from config import (
    VALID_CLAIM_STATUS, VALID_ISSUE_TYPES, VALID_SEVERITY, VALID_RISK_FLAGS,
    PARTS_MAP,
)


# Known issue families used to look up evidence_requirements.csv rows.
KNOWN_ISSUE_FAMILIES = [
    "dent or scratch",
    "crack, broken, or missing part",
    "vehicle identity or orientation",
    "screen, keyboard, or trackpad",
    "hinge, lid, corner, body, or port",
    "crushed, torn, or seal damage",
    "water, stain, or label damage",
    "contents or inner item",
]


def _parts_for(claim_object: str) -> List[str]:
    return PARTS_MAP.get(claim_object, PARTS_MAP["car"])


# ---------------------------------------------------------------------------
# Pass 1 — intent extraction (text only)
# ---------------------------------------------------------------------------

PASS1_SYSTEM = (
    "You are a damage-claim intake assistant. Your only job is to read a "
    "customer support conversation and extract what damage the customer is "
    "actually claiming. Be literal: report what the customer says, not what "
    "is true of any image. Output strict JSON matching the schema."
)


def pass1_user_prompt(user_claim: str, claim_object: str) -> str:
    return (
        f"Claim object type: {claim_object}\n\n"
        f"Conversation:\n{user_claim}\n\n"
        "Extract the claimed damage as JSON:\n"
        "- claimed_damage_description: one sentence paraphrasing the damage "
        "the customer says occurred.\n"
        "- issue_family: choose the single most relevant family from this "
        f"list: {KNOWN_ISSUE_FAMILIES}. If none fit, use 'general claim review'.\n"
        "- claimed_object_part: the specific object part the customer names "
        f"(must be one of: {_parts_for(claim_object)}). If unspecified, use "
        "'unknown'.\n"
        "Report ONLY the claim text implies; do not infer from photos."
    )


# ---------------------------------------------------------------------------
# Pass 2 — multimodal analysis
# ---------------------------------------------------------------------------

PASS2_SYSTEM = (
    "You are a meticulous damage-claim evidence reviewer. The IMAGES are the "
    "primary source of truth. The conversation tells you what to check; user "
    "history is only risk context and must NEVER override clear visual "
    "evidence. Be skeptical and concrete. Ground every conclusion in what is "
    "visible in specific images, and cite image IDs. Output strict JSON "
    "matching the schema."
)


def pass2_user_prompt(
    *,
    claim_object: str,
    user_claim: str,
    claimed_damage_description: str,
    issue_family: str,
    evidence_requirement: str,
    history_risk: bool,
    image_ids: List[str],
) -> str:
    parts = _parts_for(claim_object)
    history_line = (
        "HISTORY RISK: true (this user has elevated historical risk; this is "
        "context only — do NOT let it override the visual evidence)."
        if history_risk else
        "HISTORY RISK: false."
    )
    return (
        f"CLAIM OBJECT: {claim_object}\n"
        f"IMAGE IDs SUBMITTED: {', '.join(image_ids) if image_ids else '(none)'}\n\n"
        f"WHAT THE CUSTOMER CLAIMS:\n{user_claim}\n\n"
        f"CLAIMED DAMAGE (extracted): {claimed_damage_description}\n"
        f"ISSUE FAMILY: {issue_family}\n\n"
        f"EVIDENCE REQUIREMENT:\n{evidence_requirement}\n\n"
        f"{history_line}\n\n"
        "Analyze the attached images and decide:\n"
        f"- issue_type: one of {VALID_ISSUE_TYPES}\n"
        f"- object_part: one of {parts}\n"
        f"- claim_status: one of {VALID_CLAIM_STATUS}\n"
        f"- severity: one of {VALID_SEVERITY}\n"
        f"- risk_flags: semicolon-separated subset of {VALID_RISK_FLAGS} "
        "(do NOT include 'user_history_risk' — it is added automatically). "
        "Use 'none' if no flags apply.\n"
        "- valid_image: true if the image set is usable for automated review "
        "(not blurry/corrupt/screenshot). NOTE: a non-original image can still "
        "be clear enough to contradict a claim.\n"
        "- evidence_standard_met: true if the image set is sufficient to "
        "evaluate THIS claim (even if the answer is 'no damage').\n"
        "- supporting_image_ids: semicolon-separated subset of the submitted "
        "image IDs that support your decision; 'none' if none suffice.\n"
        "- claim_status_justification: <=2 sentences, grounded in specific "
        "image IDs.\n"
        "- evidence_standard_met_reason: <=1 sentence.\n\n"
        "Guidance:\n"
        "* supported: images clearly show the claimed damage.\n"
        "* contradicted: images clearly show no damage OR a different issue "
        "than claimed (still requires sufficient evidence).\n"
        "* not_enough_information: images cannot be used to decide.\n"
        "* Every supporting_image_id must be from the submitted list."
    )
