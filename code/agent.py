"""VLM claim-analysis agent: two-pass pipeline with retry, fallback, cache.

Implements specs/agent_contract.md. Uses the google-genai SDK with
gemini-2.5-flash at temperature=0.

Public surface used by main.py / evaluation:
    build_client()                 -> genai.Client (reads GEMINI_API_KEY)
    analyze_claim_raw(...)         -> model-output dict (Pass 1 + Pass 2)
    analyze_claim(...)             -> postprocessed dict (ready for output.csv)

The postprocessor (postprocessor.py) is the sole authority on contract
conformance; this module produces the raw model output and a safe default on
failure.
"""
import io
import json
import logging
import time
from typing import Dict, List, Tuple

from PIL import Image
from google import genai
from google.genai import errors as genai_errors, types

from config import (
    GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, SLEEP_BETWEEN_CALLS, MAX_RETRIES,
)
from cache import cache_key, cache_get, cache_set
from prompts import (
    PASS1_SYSTEM, PASS2_SYSTEM, pass1_user_prompt, pass2_user_prompt,
)
from schema import ClaimAnalysis, ClaimIntentExtraction
from image_utils import image_id_from_path

logger = logging.getLogger(__name__)

# Track real model-call counts for the operational analysis (Sprint 5).
stats = {"pass1_calls": 0, "pass2_calls": 0, "cache_hits": 0,
         "pass1_failures": 0, "pass2_failures": 0,
         "input_tokens": 0, "output_tokens": 0}


def reset_stats() -> None:
    for k in stats:
        stats[k] = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def build_client() -> genai.Client:
    """Build a genai.Client from GEMINI_API_KEY. Raises if the key is unset."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Put it in code/.env or the environment."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Retry / rate-limit helpers
# ---------------------------------------------------------------------------

def is_retryable(err: Exception) -> bool:
    """Retry on 429 (ResourceExhausted/quota) and 5xx server errors only."""
    if isinstance(err, genai_errors.APIError):
        return err.code == 429 or 500 <= err.code < 600
    if isinstance(err, genai_errors.ServerError):
        return True
    return False


def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff: multiplier 2, min SLEEP_BETWEEN_CALLS, max 60s.

    attempt is 0-based (first retry waits min). Capped at 60s per the contract.
    """
    delay = min(60.0, SLEEP_BETWEEN_CALLS * (2 ** attempt))
    time.sleep(delay)


def call_with_retry(fn, what: str = "model call"):
    """Call ``fn`` with retry on retryable errors (up to MAX_RETRIES attempts).

    Non-retryable errors propagate immediately. Raises the last error if all
    attempts fail.
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - filter below
            if not is_retryable(err):
                raise
            last_err = err
            logger.warning("%s attempt %d/%d failed (%s); retrying",
                           what, attempt + 1, MAX_RETRIES, err)
            _backoff_sleep(attempt)
    raise last_err


# ---------------------------------------------------------------------------
# Safe default (agent_contract §4)
# ---------------------------------------------------------------------------

def safe_default_analysis(history_risk: bool) -> Dict:
    """Return a contract-compliant safe default for a failed analysis."""
    flags = "manual_review_required"
    if history_risk:
        flags = "user_history_risk;manual_review_required"
    return {
        "evidence_standard_met": False,
        "evidence_standard_met_reason":
            "Model analysis unavailable; image could not be evaluated.",
        "risk_flags": flags,
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification":
            "Automated review unavailable; manual review required.",
        "supporting_image_ids": "none",
        "valid_image": False,
        "severity": "unknown",
    }


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _record_usage(resp) -> None:
    try:
        u = getattr(resp, "usage_metadata", None)
        if u is not None:
            stats["input_tokens"] += int(getattr(u, "prompt_token_count", 0) or 0)
            stats["output_tokens"] += int(getattr(u, "candidates_token_count", 0) or 0)
    except Exception:
        pass


def _extract_json(resp) -> dict:
    """Extract a JSON object from a model response (handles code fences)."""
    text = resp.text or ""
    s = text.strip()
    if s.startswith("```"):
        # strip ```json ... ``` fences
        s = s.split("```", 2)
        # s[1] holds the inner content, possibly prefixed by 'json'
        inner = s[1] if len(s) >= 2 else text
        inner = inner.strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        s = inner
    return json.loads(s)


# ---------------------------------------------------------------------------
# Pass 1 — text-only intent extraction
# ---------------------------------------------------------------------------

def _run_pass1(client, user_claim: str, claim_object: str) -> Dict:
    user_p = pass1_user_prompt(user_claim, claim_object)
    cfg = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        system_instruction=PASS1_SYSTEM,
        response_mime_type="application/json",
        response_schema=ClaimIntentExtraction,
    )

    def go():
        resp = client.models.generate_content(
            model=MODEL_NAME, contents=user_p, config=cfg,
        )
        _record_usage(resp)
        return _extract_json(resp)

    return call_with_retry(go, what="Pass 1")


# ---------------------------------------------------------------------------
# Pass 2 — multimodal analysis
# ---------------------------------------------------------------------------

def _run_pass2(client, *, claim_object, user_claim, claimed_damage_description,
               issue_family, evidence_requirement, history_risk,
               images: List[Tuple[str, Image.Image]]) -> Dict:
    image_ids = [image_id_from_path(p) for p, _ in images]
    user_p = pass2_user_prompt(
        claim_object=claim_object, user_claim=user_claim,
        claimed_damage_description=claimed_damage_description,
        issue_family=issue_family,
        evidence_requirement=evidence_requirement,
        history_risk=history_risk, image_ids=image_ids,
    )

    # Build multimodal contents: text prompt + each image part.
    contents: List = [user_p]
    img_hashes = []
    for _, img in images:
        png = _pil_to_png_bytes(img)
        from image_utils import image_hash
        img_hashes.append(image_hash(png))
        contents.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    cfg = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        system_instruction=PASS2_SYSTEM,
        response_mime_type="application/json",
        response_schema=ClaimAnalysis,
    )

    def go():
        resp = client.models.generate_content(
            model=MODEL_NAME, contents=contents, config=cfg,
        )
        _record_usage(resp)
        return _extract_json(resp)

    return call_with_retry(go, what="Pass 2")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_claim_raw(client, *, user_claim: str, claim_object: str,
                      images: List[Tuple[str, Image.Image]],
                      evidence_requirement: str,
                      history_risk: bool) -> Dict:
    """Run the two-pass pipeline and return the raw model-output dict.

    - Caches Pass 2 by a deterministic key (model, temperature, prompts, image
      hashes) so a crash/resume skips completed claims (US-006).
    - Pass 1 failure falls back to issue_family='unknown' (agent_contract §1.1).
    - Pass 2 failure after retries returns the safe default (agent_contract §4).
    """
    # ---- Pass 1 ----
    try:
        intent = _run_pass1(client, user_claim, claim_object)
        stats["pass1_calls"] += 1
        claimed_damage_description = intent.get("claimed_damage_description", "")
        issue_family = intent.get("issue_family", "unknown")
        claimed_object_part = intent.get("claimed_object_part", "unknown")
    except Exception as err:
        stats["pass1_failures"] += 1
        logger.warning("Pass 1 failed (%s); falling back to unknown family.", err)
        claimed_damage_description = user_claim[:120]
        issue_family = "unknown"
        claimed_object_part = "unknown"

    # ---- Cache key for Pass 2 ----
    from image_utils import image_hash as _ih
    img_hashes = []
    for _, img in images:
        img_hashes.append(_ih(_pil_to_png_bytes(img)))
    key = cache_key(MODEL_NAME, TEMPERATURE, PASS2_SYSTEM,
                    pass2_user_prompt(
                        claim_object=claim_object, user_claim=user_claim,
                        claimed_damage_description=claimed_damage_description,
                        issue_family=issue_family,
                        evidence_requirement=evidence_requirement,
                        history_risk=history_risk,
                        image_ids=[image_id_from_path(p) for p, _ in images],
                    ), img_hashes)

    cached = cache_get(key)
    if cached is not None:
        stats["cache_hits"] += 1
        return cached

    # ---- Pass 2 ----
    try:
        out = _run_pass2(
            client, claim_object=claim_object, user_claim=user_claim,
            claimed_damage_description=claimed_damage_description,
            issue_family=issue_family,
            evidence_requirement=evidence_requirement,
            history_risk=history_risk, images=images,
        )
        stats["pass2_calls"] += 1
        cache_set(key, out)
        return out
    except Exception as err:
        stats["pass2_failures"] += 1
        logger.warning("Pass 2 failed (%s); returning safe default.", err)
        safe = safe_default_analysis(history_risk)
        cache_set(key, safe)
        return safe


def analyze_claim(client, *, user_claim, claim_object, image_paths: str,
                  evidence_requirement: str, history_risk: bool) -> Dict:
    """Convenience wrapper: load images from paths, run pipeline, postprocess.

    Returns a postprocessor-validated dict ready for output.csv.
    """
    from image_utils import parse_image_paths, load_images
    from postprocessor import postprocess

    rel_paths = parse_image_paths(image_paths)
    images = load_images(rel_paths)
    raw = analyze_claim_raw(
        client, user_claim=user_claim, claim_object=claim_object,
        images=images, evidence_requirement=evidence_requirement,
        history_risk=history_risk,
    )
    return postprocess(raw, claim_object, image_paths, history_risk)
