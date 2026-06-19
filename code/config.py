"""Central configuration and allowed-value constants.

Single source of runtime constants. The allowed-value lists here mirror
``specs/data_contract.md``; tests in ``tests/test_schema_contract.py`` pin
them. Do not mutate these at runtime.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Secrets (env-only, never hardcoded) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Model / API behaviour ---
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0
# Inter-call throttle. With a single key this must be ~13s to stay under the
# 5 RPM free-tier limit. With a multi-key pool each key fires rarely, so this
# can be lowered (env-overridable) to speed up large runs.
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_BETWEEN_CALLS", "13.0"))
MAX_RETRIES = 5

# --- Filesystem layout ---
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_CSV = DATASET_DIR / "evidence_requirements.csv"
IMAGES_DIR = DATASET_DIR / "images"
OUTPUT_CSV = BASE_DIR / "output.csv"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- Allowed enum values (see specs/data_contract.md §4) ---
VALID_CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

VALID_ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
]

VALID_SEVERITY = ["none", "low", "medium", "high", "unknown"]

VALID_RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
]

CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown"
]
LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown"
]
PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown"
]
PARTS_MAP = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

# --- Output column order (specs/data_contract.md §3.1) ---
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity"
]
