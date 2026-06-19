"""Sprint 5 — Consistency validator tests.

Written FIRST (RED). The validator detects impossible combinations and
either force-fixes them or signals that a re-run is needed. Per the user's
choice, the pipeline re-runs Pass 2 ONCE on inconsistency, then force-fixes.

Data-grounded note (Sprint 3): valid_image=false does NOT force
evidence_standard_met=false (see user_008). A configurable flag
STRICT_VALID_IMAGE_IMPLIES_EVIDENCE enforces the stricter rule but defaults
to False to maximize exact-match accuracy.
"""
import pytest

from validator import find_inconsistencies, force_fix, STRICT_VALID_IMAGE_IMPLIES_EVIDENCE


def _analysis(**over):
    a = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "clear",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "front_bumper",
        "claim_status": "supported",
        "claim_status_justification": "img_1 shows a dent.",
        "supporting_image_ids": "img_1",
        "valid_image": True,
        "severity": "medium",
    }
    a.update(over)
    return a


class TestInconsistencyDetection:
    def test_clean_supported_is_consistent(self):
        assert find_inconsistencies(_analysis()) == []

    def test_supported_with_issue_none_is_inconsistent(self):
        inc = find_inconsistencies(_analysis(issue_type="none"))
        assert any("issue_type" in i or "supported" in i.lower() for i in inc)

    def test_supported_with_no_supporting_ids_is_inconsistent(self):
        inc = find_inconsistencies(_analysis(supporting_image_ids="none"))
        assert len(inc) > 0

    def test_supported_with_invalid_image_is_inconsistent(self):
        inc = find_inconsistencies(_analysis(valid_image=False))
        assert len(inc) > 0

    def test_supported_with_evidence_false_is_inconsistent(self):
        inc = find_inconsistencies(_analysis(evidence_standard_met=False))
        assert len(inc) > 0

    def test_high_severity_with_issue_none_is_inconsistent(self):
        inc = find_inconsistencies(_analysis(claim_status="contradicted",
                                             issue_type="none", severity="high"))
        assert any("severity" in i.lower() for i in inc)

    def test_valid_image_false_evidence_true_is_NOT_inconsistent_by_default(self):
        # user_008 pattern — must NOT be flagged with the default flag
        inc = find_inconsistencies(_analysis(
            claim_status="contradicted", valid_image=False,
            evidence_standard_met=True, issue_type="broken_part",
            supporting_image_ids="img_1", severity="high"))
        assert not any("evidence_standard_met" in i.lower() for i in inc)


class TestStrictFlagDefault:
    def test_strict_flag_defaults_false(self):
        assert STRICT_VALID_IMAGE_IMPLIES_EVIDENCE is False


class TestForceFix:
    def test_supported_issue_none_demotes_status(self):
        out = force_fix(_analysis(issue_type="none"))
        assert out["claim_status"] != "supported"

    def test_supported_no_supporting_demotes_status(self):
        out = force_fix(_analysis(supporting_image_ids="none"))
        assert out["claim_status"] != "supported"

    def test_high_severity_issue_none_severity_unknown(self):
        out = force_fix(_analysis(claim_status="contradicted",
                                  issue_type="none", severity="high"))
        assert out["severity"] == "unknown"

    def test_force_fix_never_returns_supported_when_inconsistent(self):
        # any inconsistent supported row must be demoted
        for over in [
            {"issue_type": "none"},
            {"supporting_image_ids": "none"},
            {"valid_image": False},
            {"evidence_standard_met": False},
        ]:
            out = force_fix(_analysis(**over))
            assert out["claim_status"] != "supported", over

    def test_force_fix_preserves_valid_rows(self):
        clean = _analysis()
        out = force_fix(clean)
        assert out["claim_status"] == "supported"
        assert out["issue_type"] == "dent"

    def test_high_severity_demoted_for_scratch_dent_crack(self):
        """Data-grounded: scratch/dent/crack are never 'high' in labels."""
        for issue in ("scratch", "dent", "crack"):
            out = force_fix(_analysis(issue_type=issue, severity="high"))
            assert out["severity"] == "medium", issue

    def test_high_severity_kept_for_broken_part(self):
        out = force_fix(_analysis(issue_type="broken_part", severity="high"))
        assert out["severity"] == "high"
