"""API key pool with rotation across daily quota limits.

The Gemini free tier is 20 requests/day per project+model per key. When a
key's DAILY quota is exhausted, rotate to the next key. Transient per-minute
429s are handled by the normal retry path, not by rotation.

Keys are read from env vars GEMINI_API_KEY (primary) and GEMINI_API_KEY_2,
GEMINI_API_KEY_3, ... (additional). All loaded keys are gitignored secrets.
"""
import os
from typing import List, Optional

from google import genai


class KeyExhausted(RuntimeError):
    """All keys in the pool have exhausted their daily quota."""


def is_daily_quota_429(err) -> bool:
    """True if err is a 429 caused by the per-DAY (not per-minute) quota."""
    if err is None:
        return False
    try:
        code = getattr(err, "code", None)
        if code != 429:
            return False
    except Exception:
        return False
    # Inspect the error details for the PerDay quota id.
    text = repr(err).lower()
    return ("perday" in text) or ("per_day" in text)


class KeyPool:
    """Holds genai clients built from a list of API keys; rotates on exhaustion.

    A key that hits the DAILY quota (PerDay) is marked permanently dead for the
    process lifetime (it won't recover for ~24h). Transient per-minute 429s and
    503s are handled by the retry layer, not by permanent rotation. ``reset()``
    returns the cursor to the first non-dead key, so each new call/claim can
    retry keys that were only transiently rate-limited.
    """

    def __init__(self, keys: List[str]):
        self._keys = [k for k in keys if k]
        self._clients = [genai.Client(api_key=k) for k in self._keys]
        self._idx = 0
        self._dead: set = set()  # indices of daily-exhausted keys

    def has_key(self) -> bool:
        # Any non-dead key remaining?
        return any(i not in self._dead for i in range(len(self._clients)))

    def current_index(self) -> int:
        return self._idx

    def current(self):
        if not self.has_key():
            raise KeyExhausted("no API keys remain with available daily quota")
        if self._idx >= len(self._clients) or self._idx in self._dead:
            self.reset()  # rewind to the first live key
        if self._idx >= len(self._clients):
            raise KeyExhausted("no API keys remain with available daily quota")
        return self._clients[self._idx]

    def mark_dead(self, idx: int) -> None:
        """Mark a key's DAILY quota permanently exhausted for this process."""
        self._dead.add(idx)

    def is_dead(self, idx: int) -> bool:
        return idx in self._dead

    def rotate(self) -> None:
        """Advance to the next non-dead key (for transient failures)."""
        self._idx += 1
        while self._idx < len(self._clients) and self._idx in self._dead:
            self._idx += 1

    def reset(self) -> None:
        """Rewind to the first non-dead key. Call at the start of each claim."""
        self._idx = 0
        while self._idx < len(self._clients) and self._idx in self._dead:
            self._idx += 1

    def __len__(self) -> int:
        return len(self._clients)

    def live_count(self) -> int:
        return len(self._clients) - len(self._dead)


def load_keys_from_env() -> List[str]:
    """Load GEMINI_API_KEY plus GEMINI_API_KEY_2, _3, ... from the environment."""
    keys: List[str] = []
    primary = os.getenv("GEMINI_API_KEY", "")
    if primary:
        keys.append(primary)
    i = 2
    while True:
        k = os.getenv(f"GEMINI_API_KEY_{i}", "")
        if not k:
            break
        keys.append(k)
        i += 1
    return keys


def build_pool() -> KeyPool:
    """Build a KeyPool from env vars; raises if no key is configured."""
    keys = load_keys_from_env()
    if not keys:
        raise RuntimeError(
            "No GEMINI_API_KEY set. Put it in code/.env "
            "(additional keys as GEMINI_API_KEY_2, _3, ...)."
        )
    return KeyPool(keys)
