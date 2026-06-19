"""Deterministic on-disk cache for VLM model calls.

Keys are derived from (model, temperature, prompt, image hashes) so that a
cache hit is reproducible and crash-safe (US-006: resume without reprocessing
completed claims). Cache store lives under code/.cache/ (git-ignored).
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from config import CACHE_DIR


def _key_str(*parts: Any) -> str:
    return "\u0001".join(str(p) for p in parts)


def cache_key(model: str, temperature, system_prompt: str, user_prompt: str,
              image_hashes) -> str:
    """Compute a stable SHA-256 cache key for a model call."""
    h = hashlib.sha256()
    h.update(_key_str(model, temperature, system_prompt, user_prompt).encode("utf-8"))
    for ih in sorted(image_hashes or []):
        h.update(ih.encode("utf-8"))
    return h.hexdigest()


def cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_get(key: str) -> Optional[Any]:
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_set(key: str, value: Any) -> None:
    p = cache_path(key)
    try:
        p.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Cache is best-effort; never crash the pipeline on a write failure.
        pass


def get_or_compute(key: str, compute):
    """Return cached value for ``key`` or compute, cache, and return it."""
    cached = cache_get(key)
    if cached is not None:
        return cached, True   # (value, was_cache_hit)
    value = compute()
    cache_set(key, value)
    return value, False
