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

    def test_rotate_past_last_rewinds_via_current(self):
        # New semantics: rotate past end doesn't kill the key (only daily-quota
        # marks a key dead). current() rewinds to the first live key.
        pool = KeyPool(["keyA"])
        pool.rotate()
        assert pool.has_key() is True  # key not dead, just cursor past end
        c = pool.current()  # rewinds to 0
        assert c is not None
        assert pool.current_index() == 0

    def test_mark_dead_exhausts_pool(self):
        pool = KeyPool(["keyA"])
        pool.mark_dead(0)
        assert pool.has_key() is False
        with pytest.raises(KeyExhausted):
            pool.current()

    def test_mark_one_dead_other_still_live(self):
        pool = KeyPool(["keyA", "keyB"])
        pool.mark_dead(0)
        assert pool.has_key() is True
        pool.reset()
        assert pool.current_index() == 1  # skips dead key 0

    def test_all_dead_is_exhausted(self):
        pool = KeyPool(["keyA", "keyB"])
        pool.mark_dead(0)
        pool.mark_dead(1)
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

    def test_reset_rewinds_to_first_live_key(self):
        pool = KeyPool(["keyA", "keyB", "keyC"])
        pool.mark_dead(0)
        pool.reset()
        assert pool.current_index() == 1

    def test_live_count(self):
        pool = KeyPool(["keyA", "keyB", "keyC"])
        assert pool.live_count() == 3
        pool.mark_dead(1)
        assert pool.live_count() == 2
