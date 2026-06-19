"""Image loading and path/ID utilities.

Implements specs/data_contract.md §2.5 (image path format & image IDs) and
supports scenarios 5 & 9 of behavior_spec.md (missing images, per-image IDs).
Image paths are relative to the dataset root (config.DATASET_DIR).
"""
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from config import DATASET_DIR


def parse_image_paths(image_paths_field: str) -> List[str]:
    """Split a semicolon-separated image_paths cell into cleaned paths."""
    if not image_paths_field:
        return []
    return [p.strip() for p in str(image_paths_field).split(";") if p.strip()]


def image_id_from_path(path: str) -> str:
    """Return the image ID: filename without extension.

    e.g. 'images/test/case_001/img_1.jpg' -> 'img_1'.
    """
    return Path(path).stem


def _resolve(rel_path: str) -> Path:
    return DATASET_DIR / rel_path


def load_image(rel_path: str) -> Optional[Image.Image]:
    """Load a single image by path relative to the dataset root.

    Returns None if the file is missing or unreadable (Scenario 5) — never
    raises. Converts to RGB so downstream model clients get a consistent mode.
    """
    p = _resolve(rel_path)
    try:
        if not p.exists():
            return None
        img = Image.open(p)
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        return None


def load_images(rel_paths: List[str]) -> List[Tuple[str, Image.Image]]:
    """Load all loadable images. Returns list of (rel_path, PIL.Image).

    Missing/unreadable images are silently skipped (Scenario 5).
    """
    out = []
    for rel in rel_paths:
        img = load_image(rel)
        if img is not None:
            out.append((rel, img))
    return out


def image_hash(image_bytes: bytes) -> str:
    """Stable hex hash of raw image bytes, used for cache keys."""
    return hashlib.sha256(image_bytes).hexdigest()
