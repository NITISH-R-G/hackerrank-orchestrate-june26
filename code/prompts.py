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

# Strategy A uses the same reviewer persona; only the prompt assembly differs.
SINGLE_PASS_SYSTEM = PASS2_SYSTEM


def single_pass_user_prompt(
    *,
    claim_object: str,
    user_claim: str,
    evidence_requirement: str,
    history_risk: bool,
    image_ids: List[str],
) -> str:
    """Strategy A: one call does intent extraction + analysis together.

    Same strict-enum/anti-hallucination body as pass2_user_prompt, but the
    model must itself figure out the claimed damage from the raw conversation
    (no extracted intent provided).
    """
    parts = _parts_for(claim_object)
    history_line = (
        "HISTORY RISK: true (context only — do NOT let it override the visual "
        "evidence)." if history_risk else "HISTORY RISK: false."
    )
    return (
        f"CLAIM OBJECT: {claim_object}\n"
        f"IMAGE IDs SUBMITTED: {', '.join(image_ids) if image_ids else '(none)'}\n\n"
        f"CUSTOMER CONVERSATION:\n{user_claim}\n\n"
        f"EVIDENCE REQUIREMENT:\n{evidence_requirement}\n\n"
        f"{history_line}\n\n"
        "First, determine what damage the customer is claiming from the "
        "conversation. Then analyze the attached images and decide the "
        "following fields.\n\n"
        "=== object_part (STRICT ENUM — never invent) ===\n"
        f"Pick EXACTLY one from: {parts}\n"
        "Map common descriptions to the canonical label BEFORE output:\n"
        "  'back bumper'/'rear bumper' -> rear_bumper; 'front bumper' -> front_bumper;\n"
        "  'rear side panel'/'side panel' -> quarter_panel; 'rear windshield'/'back glass' -> windshield;\n"
        "  'back light'/'tail light' -> taillight; 'head light' -> headlight; 'mirror' -> side_mirror;\n"
        "  'trunk' -> body. If no clear match, output 'unknown'.\n\n"
        "=== issue_type (STRICT ENUM): one of [dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown] ===\n"
        "KEY DISTINCTIONS: crack=a single intact fissure line (stone-chip spread = CRACK); glass_shatter=glass broken into many pieces/spider-webbed; broken_part=non-glass component snapped/detached; "
        "dent=metal deformation; scratch=surface mark. 'none' ONLY if part visible+undamaged; do NOT invent damage. Map: 'screen crack'->crack; 'water spill'->water_damage.\n\n"
        "=== severity (STRICT, deterministic — be conservative) ===\n"
        "  none=no damage (only if issue_type=none);\n"
        "  low=minor cosmetic (light scratch, small/shallow dent) — default for a single scratch or small dent;\n"
        "  medium=clearly visible localized damage — one distinct dent, one clear crack, one isolated broken component;\n"
        "  high=SEVERE ONLY — large deformation across a panel, shattered glass, MULTIPLE damaged components, or function-impairing damage. A single dent/scratch is almost NEVER high.\n"
        "  unknown=cannot determine. When unsure prefer the lower of two adjacent levels.\n\n"
        f"=== claim_status (STRICT ENUM): one of {VALID_CLAIM_STATUS} ===\n"
        f"=== risk_flags (semicolon-separated subset of {VALID_RISK_FLAGS}; exclude 'user_history_risk'; 'none' if none) ===\n\n"
        "- valid_image: true unless blurry/corrupt (a screenshot can still be clear enough to contradict).\n"
        "- evidence_standard_met: true if the set is sufficient to evaluate THIS claim.\n"
        "- supporting_image_ids: subset of submitted IDs; 'none' only if none suffice.\n"
        "- claim_status_justification: <=2 sentences citing image IDs.\n"
        "- evidence_standard_met_reason: <=1 sentence.\n\n"
        "MULTI-IMAGE CONSISTENCY (CRITICAL): When MULTIPLE images are submitted, "
        "first verify they depict the SAME object/vehicle. If a close-up shows "
        "damage but a full-view image appears to be a DIFFERENT object (color, "
        "model, plate, setting), the set does NOT establish identity: "
        "evidence_standard_met=false, claim_status=not_enough_information, "
        "risk_flags must include 'wrong_object' AND 'claim_mismatch', severity=unknown.\n\n"
        "DECISION RULES: supported=images show the claimed damage; contradicted=images show no damage OR a different issue (needs sufficient evidence); not_enough_information=cannot decide OR cross-image identity fails.\n"
        "ANTI-HALLUCINATION: report ONLY visible damage; never invent parts/issues/severity; when uncertain prefer not_enough_information; history risk never changes claim_status."
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
        "Analyze the attached images and decide the following fields.\n\n"
        "=== object_part (STRICT ENUM — never invent) ===\n"
        f"Pick EXACTLY one from: {parts}\n"
        "Map common descriptions to the canonical label BEFORE output:\n"
        "  'back bumper' / 'rear bumper' -> rear_bumper;\n"
        "  'front bumper' -> front_bumper;\n"
        "  'rear side panel' / 'side panel' -> quarter_panel;\n"
        "  'rear windshield' / 'back glass' -> windshield;\n"
        "  'back light' / 'tail light' -> taillight;\n"
        "  'head light' -> headlight;\n"
        "  'mirror' -> side_mirror;\n"
        "  'trunk' -> body (unless clearly the bumper).\n"
        "If the visible part does not clearly match any label, output "
        "'unknown'. NEVER output a word that is not in the list above.\n\n"
        "=== issue_type (STRICT ENUM — never invent) ===\n"
        f"Pick EXACTLY one from: {VALID_ISSUE_TYPES}\n"
        "CRITICAL distinctions (choose carefully):\n"
        "  - crack vs glass_shatter: 'crack' = a single crack LINE that is still "
        "intact (a fissure running through glass/screen/plastic). 'glass_shatter' = "
        "glass is SHATTERED into many pieces / spider-webbed / splintered (multiple "
        "intersecting breaks). A stone-chip with a spreading line is a CRACK, not "
        "shatter. Default to 'crack' unless the glass is clearly broken into pieces.\n"
        "  - broken_part: a non-glass component is broken/snapped/detached "
        "(mirror housing cracked off, hinge snapped, key missing-stem). Use for "
        "plastic/metal parts, NOT for intact glass (that is crack/shatter).\n"
        "  - dent vs scratch: dent = deformation/indentation in metal; scratch = "
        "surface mark/gouge without deformation.\n"
        "  - stain vs water_damage: stain = a visible mark/discoloration; "
        "water_damage = active liquid/water signs (warping, mineral residue).\n"
        "Map aliases: 'screen crack'->crack; 'water spill'->water_damage. "
        "Use 'none' ONLY when the part is visible and UNDAMAGED; 'unknown' ONLY "
        "when it cannot be determined. Do NOT invent damage that isn't clearly "
        "visible — if the part looks fine, the answer is 'none'.\n\n"
        "=== severity (STRICT, deterministic definitions — be conservative) ===\n"
        f"Pick EXACTLY one from: {VALID_SEVERITY}\n"
        "  - none: no damage present (only valid when issue_type=none).\n"
        "  - low: minor cosmetic damage — a light surface scratch, a small/shallow "
        "dent, a faint mark. Default for a single scratch or small dent.\n"
        "  - medium: clearly visible localized damage — one distinct dent, one "
        "clear crack, or one isolated broken component. Default for a single "
        "clear dent or crack.\n"
        "  - high: SEVERE damage ONLY — large deformation across a panel, "
        "shattered glass, MULTIPLE damaged components, or damage that clearly "
        "impairs function. A single dent or scratch is almost NEVER high; do "
        "not use high unless the damage is clearly major/extensive.\n"
        "  - unknown: cannot determine from the images. Do NOT guess.\n"
        "When unsure between low and medium, prefer low; between medium and "
        "high, prefer medium.\n\n"
        f"=== claim_status (STRICT ENUM): one of {VALID_CLAIM_STATUS} ===\n"
        f"=== risk_flags (semicolon-separated subset of {VALID_RISK_FLAGS}) ===\n"
        "Do NOT include 'user_history_risk' (added automatically). Use 'none' "
        "if no flags apply.\n\n"
        "- valid_image: true if the image set is usable for automated review "
        "(not blurry/corrupt). NOTE: a non-original/screenshot image can still "
        "be clear enough to contradict a claim — set valid_image=false for "
        "screenshots/non-original images, but that does NOT prevent "
        "evidence_standard_met=true.\n"
        "- evidence_standard_met: true if the image set is sufficient to "
        "evaluate THIS claim (even if the answer is 'no damage' or 'wrong "
        "damage'). False only when you genuinely cannot evaluate the claim.\n"
        "- supporting_image_ids: semicolon-separated subset of the submitted "
        "image IDs that support your decision; 'none' only if none suffice.\n"
        "- claim_status_justification: <=2 sentences, grounded in specific "
        "image IDs.\n"
        "- evidence_standard_met_reason: <=1 sentence.\n\n"
        "=== MULTI-IMAGE CONSISTENCY (CRITICAL) ===\n"
        "When MULTIPLE images are submitted, first verify they depict the SAME "
        "object/vehicle. If a close-up shows damage but a full-view image "
        "appears to be a DIFFERENT object (different color, model, license "
        "plate, or setting), the image set does NOT establish identity:\n"
        "  -> evidence_standard_met = false\n"
        "  -> claim_status = not_enough_information\n"
        "  -> risk_flags must include 'wrong_object' AND 'claim_mismatch'\n"
        "  -> severity = unknown\n"
        "This vehicle-identity / cross-image check applies to cars, laptops, "
        "and packages alike.\n\n"
        "=== DECISION RULES (follow exactly) ===\n"
        "* supported: images clearly show the claimed damage type on the "
        "claimed part.\n"
        "* contradicted: images clearly show NO damage OR a different issue "
        "than claimed (requires sufficient evidence to be sure).\n"
        "* not_enough_information: images cannot be used to decide, OR the "
        "image set fails cross-image consistency (different objects).\n\n"
        "=== ANTI-HALLUCINATION ===\n"
        "* Report ONLY what is visible in the images. Never invent damage, "
        "parts, issue types, or severity.\n"
        "* If uncertain between supported/contradicted, prefer "
        "not_enough_information rather than guessing.\n"
        "* Every supporting_image_id MUST come from the submitted list.\n"
        "* History risk is context only; it must NEVER change claim_status."
    )
