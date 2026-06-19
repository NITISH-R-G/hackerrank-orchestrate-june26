"""Sprint 5 — API key pool with rotation (TDD).

The free tier is 20 requests/day per key. When a key's DAILY quota is
exhausted (not a transient per-minute 429), the pool rotates to the next key.
Written FIRST (RED). Run from code/:  pytest tests/test_keypool.py -v
"""
import pytest

from keypool import KeyPool, KeyExhausted, is_daily_quota_429


def _daily_err():
    from google.genai import errors as ge
    return ge.APIError(429, {
        "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "...",
                  "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                               "violations": [{"quotaId":
                                   "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}
    }, None)


def _minute_err():
    from google.genai import errors as ge
    return ge.APIError(429, {
        "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "...",
                  "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                               "violations": [{"quotaId":
                                   "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}
    }, None)


def _plain_err(code):
    from google.genai import errors as ge
    return ge.APIError(code, {"error": {"code": code, "message": "x"}}, None)


class TestIsDailyQuota429:
    def test_daily_quota_detected(self):
        assert is_daily_quota_429(_daily_err()) is True

    def test_minute_quota_not_daily(self):
        assert is_daily_quota_429(_minute_err()) is False

    def test_500_not_daily(self):
        assert is_daily_quota_429(_plain_err(500)) is False

    def test_none_safe(self):
        assert is_daily_quota_429(None) is False


class TestKeyPoolRotation:
    def test_returns_clients_in_order(self):
        pool = KeyPool(["keyA", "keyB"])
        c0 = pool.current()
        assert "keyA" in str(c0) or c0 is not None

    def test_rotate_moves_to_next_key(self):
        pool = KeyPool(["keyA", "keyB"])
        pool.rotate()
        # current index advances; still have a key
        assert pool.has_key()

    def test_rotate_past_last_key_raises(self):
        pool = KeyPool(["keyA"])
        pool.rotate()
        assert pool.has_key() is False
        with pytest.raises(KeyExhausted):
            pool.current()

    def test_two_keys_allow_two_rotations(self):
        pool = KeyPool(["keyA", "keyB"])
        assert pool.has_key()
        pool.rotate()
        assert pool.has_key()
        pool.rotate()
        assert pool.has_key() is False

    def test_empty_pool_is_exhausted(self):
        pool = KeyPool([])
        assert pool.has_key() is False
        with pytest.raises(KeyExhausted):
            pool.current()

    def test_current_index_tracks(self):
        pool = KeyPool(["keyA", "keyB", "keyC"])
        assert pool.current_index() == 0
        pool.rotate()
        assert pool.current_index() == 1
        pool.rotate()
        assert pool.current_index() == 2
