"""pytest configuration: register the 'live' marker.

'live' tests make real Gemini API calls (need GEMINI_API_KEY + network).
They are skipped by default; run them explicitly:

    pytest -m live              # live only
    pytest -m "not live"        # deterministic only (the default)
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: makes real Gemini API calls (needs GEMINI_API_KEY)"
    )


def pytest_runtest_setup(item):
    if "live" in {m.name for m in item.iter_markers()}:
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set; skipping live test")
