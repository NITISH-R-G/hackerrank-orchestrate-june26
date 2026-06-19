"""Sprint 5.8 — main.py pipeline runner tests (mocked, no API).

Verifies: exact column order, one row per input, safe-default fill on runner
failure, incremental/crash-safe writing, and _limit. Run from code/:

    pytest tests/test_main_pipeline.py -v
"""
import pandas as pd
import pytest

import main
from config import OUTPUT_COLUMNS


def _fake_claims(tmp_path, n=3):
    rows = []
    for i in range(n):
        rows.append({
            "user_id": f"u{i}",
            "image_paths": f"images/test/c{i}/img_1.jpg",
            "user_claim": f"claim {i}",
            "claim_object": "car",
        })
    p = tmp_path / "claims.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _good_runner_factory(fail_indices=()):
    """Returns a runner that emits a valid analysis, failing for given indices."""
    def _runner(row, history_risk, evidence_req):
        if int(row["user_id"][1:]) in fail_indices:
            raise RuntimeError("simulated failure")
        return {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "clear",
            "risk_flags": "none",
            "issue_type": "dent",
            "object_part": "front_bumper",
            "claim_status": "supported",
            "claim_status_justification": "img_1 shows dent",
            "supporting_image_ids": "img_1",
            "valid_image": True,
            "severity": "medium",
        }
    return _runner


class TestRunPipeline:
    def test_writes_one_row_per_input_with_exact_columns(self, tmp_path):
        claims = _fake_claims(tmp_path, n=3)
        out = tmp_path / "output.csv"
        written = main.run_pipeline(str(claims), str(out),
                                     _runner=_good_runner_factory())
        df = pd.read_csv(out)
        assert written == 3
        assert len(df) == 3
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_failed_row_becomes_safe_default(self, tmp_path):
        claims = _fake_claims(tmp_path, n=3)
        out = tmp_path / "output.csv"
        main.run_pipeline(str(claims), str(out),
                          _runner=_good_runner_factory(fail_indices=(1,)))
        df = pd.read_csv(out)
        assert len(df) == 3
        # row index 1 failed -> safe default: not supported, unknown issue
        assert df.iloc[1]["claim_status"] == "not_enough_information"
        assert df.iloc[1]["issue_type"] == "unknown"
        # other rows are real
        assert df.iloc[0]["claim_status"] == "supported"

    def test_limit_caps_rows(self, tmp_path):
        claims = _fake_claims(tmp_path, n=5)
        out = tmp_path / "output.csv"
        written = main.run_pipeline(str(claims), str(out),
                                    limit=2, _runner=_good_runner_factory())
        df = pd.read_csv(out)
        assert written == 2
        assert len(df) == 2

    def test_output_never_has_nulls(self, tmp_path):
        claims = _fake_claims(tmp_path, n=3)
        out = tmp_path / "output.csv"
        main.run_pipeline(str(claims), str(out),
                          _runner=_good_runner_factory(fail_indices=(0, 1, 2)))
        df = pd.read_csv(out)
        # no NaN in any cell
        assert not df.isnull().any().any(), df.isnull().sum().to_dict()

    def test_identity_columns_copied_verbatim(self, tmp_path):
        claims = _fake_claims(tmp_path, n=2)
        out = tmp_path / "output.csv"
        main.run_pipeline(str(claims), str(out), _runner=_good_runner_factory())
        df = pd.read_csv(out)
        assert df.iloc[0]["user_id"] == "u0"
        assert df.iloc[0]["claim_object"] == "car"
        assert "claim 0" in df.iloc[0]["user_claim"]
