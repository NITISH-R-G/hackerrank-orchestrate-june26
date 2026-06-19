"""Deterministic categorical normalizer.

Maps common free-form descriptions to the official enums BEFORE fuzzy-matching,
so 'back bumper' -> 'rear_bumper' is deterministic rather than reliant on
difflib proximity. This is the first stage of post-processing; the
postprocessor then enforces object-bound parts and consistency invariants.

Specs: problem_statement.md allowed values + Sprint 5 normalization rules.
"""
import difflib

from config import (
    VALID_CLAIM_STATUS, VALID_ISSUE_TYPES, VALID_SEVERITY, VALID_RISK_FLAGS,
    PARTS_MAP,
)


# Explicit alias -> canonical maps. Keys are matched case-insensitively after
# lowercasing + collapsing spaces to underscores.
NORMALIZER_ALIASES = {
    "object_part": {
        # car
        "back bumper": "rear_bumper",
        "rear bumper": "rear_bumper",
        "front bumper": "front_bumper",
        "rear side panel": "quarter_panel",
        "side panel": "quarter_panel",
        "rear windshield": "windshield",
        "back glass": "windshield",
        "back light": "taillight",
        "tail light": "taillight",
        "head light": "headlight",
        "mirror": "side_mirror",
        "side mirror": "side_mirror",
        "trunk": "body",
        "bonnet": "hood",
        "wing mirror": "side_mirror",
        # laptop
        "laptop screen": "screen",
        "display": "screen",
        "touchpad": "trackpad",
        "clamshell": "lid",
        "edge": "corner",
        "usb port": "port",
        "charging port": "port",
        "bottom": "base",
        # package
        "box corner": "package_corner",
        "box side": "package_side",
        "flap": "seal",
        "sticker": "label",
        "inside": "contents",
        "product": "item",
    },
    "issue_type": {
        "screen crack": "crack",
        "cracked screen": "crack",
        "hairline crack": "crack",
        "water spill": "water_damage",
        "liquid damage": "water_damage",
        "spill": "water_damage",
        "shattered glass": "glass_shatter",
        "shattered": "glass_shatter",
        "smashed": "broken_part",
        "snapped": "broken_part",
        "ripped": "torn_packaging",
        "torn": "torn_packaging",
        "crumpled": "crushed_packaging",
        "dented": "dent",
        "gouge": "scratch",
        "scuff": "scratch",
        "discoloration": "stain",
        "mark": "stain",
    },
    "severity": {
        "minor": "low",
        "cosmetic": "low",
        "small": "low",
        "slight": "low",
        "moderate": "medium",
        "medium damage": "medium",
        "severe": "high",
        "major": "high",
        "heavy": "high",
        "extensive": "high",
        "large": "high",
        "bad": "high",
    },
}


def _key(s):
    return str(s).strip().lower()


def _canonical_or_fuzzy(value, allowed, fallback, aliases=None):
    """alias -> exact -> difflib closest -> fallback."""
    k = _key(value)
    if not k:
        return fallback
    if aliases and k in aliases:
        return aliases[k]
    # allow either spaces or underscores in the input
    k_und = k.replace(" ", "_")
    if k_und in allowed:
        return k_und
    if k in allowed:
        return k
    matches = difflib.get_close_matches(k_und, allowed, n=1, cutoff=0.6)
    if not matches:
        matches = difflib.get_close_matches(k, allowed, n=1, cutoff=0.6)
    return matches[0] if matches else fallback


def normalize_object_part(value, claim_object):
    allowed = PARTS_MAP.get(claim_object, PARTS_MAP["car"])
    out = _canonical_or_fuzzy(value, allowed, "unknown", NORMALIZER_ALIASES["object_part"])
    if out not in allowed:
        return "unknown"
    return out


def normalize_issue_type(value):
    return _canonical_or_fuzzy(value, VALID_ISSUE_TYPES, "unknown",
                               NORMALIZER_ALIASES["issue_type"])


def normalize_severity(value):
    return _canonical_or_fuzzy(value, VALID_SEVERITY, "unknown",
                               NORMALIZER_ALIASES["severity"])


def normalize_claim_status(value):
    return _canonical_or_fuzzy(value, VALID_CLAIM_STATUS,
                               "not_enough_information")


def normalize_risk_flags(value):
    if not value:
        return "none"
    tokens = [t.strip() for t in str(value).split(";") if t.strip()]
    out = []
    seen = set()
    for t in tokens:
        tl = t.lower()
        if tl == "none":
            continue
        # alias-free; just validate against allowed
        m = difflib.get_close_matches(tl, VALID_RISK_FLAGS, n=1, cutoff=0.8)
        canonical = m[0] if m else None
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return ";".join(out) if out else "none"
