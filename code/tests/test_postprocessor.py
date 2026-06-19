"""Sprint 3 — Postprocessor tests.

Written FIRST (RED) against the corrected specs/data_contract.md §5
invariants and specs/behavior_spec.md cross-cutting rules. The postprocessor
is the sole deterministic guarantor of contract conformance. Run from code/:

    pytest tests/test_postprocessor.py -v

Key data-grounded note (Scenario 7, refined): `valid_image=false` does NOT
force `evidence_standard_met=false` — see user_008 in sample_claims.csv.
"""
import pytest

from postprocessor import postprocess


def _base_analysis(**overrides):
    """A clean, fully-valid model output to mutate per test."""
    out = {
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "Image is clear.",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "front_bumper",
        "claim_status": "supported",
        "claim_status_justification": "img_1 shows a dent.",
        "supporting_image_ids": "img_1",
        "valid_image": True,
        "severity": "medium",
    }
    out.update(overrides)
    return out


# ---- Rule 1: enum normalization + fuzzy fix (difflib) ----

class TestEnumNormalization:
    def test_lowercases_and_strips(self):
        r = postprocess(_base_analysis(claim_status=" SUPPORTED "),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["claim_status"] == "supported"

    def test_fuzzy_fixes_near_miss(self):
        # 'front_bmpr' is a near-miss of 'front_bumper'
        r = postprocess(_base_analysis(object_part="front_bmpr"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["object_part"] == "front_bumper"

    def test_unfixable_issue_type_falls_back_to_unknown(self):
        r = postprocess(_base_analysis(issue_type="explode"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["issue_type"] == "unknown"

    def test_unfixable_severity_falls_back_to_unknown(self):
        r = postprocess(_base_analysis(severity="huge"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["severity"] == "unknown"

    def test_unfixable_claim_status_falls_back_to_nei(self):
        r = postprocess(_base_analysis(claim_status="maybe"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["claim_status"] == "not_enough_information"


# ---- Rule 2: object-bound part validation ----

class TestObjectBoundPart:
    def test_car_part_on_car_row_kept(self):
        r = postprocess(_base_analysis(object_part="door"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["object_part"] == "door"

    def test_laptop_part_on_car_row_mapped_to_unknown(self):
        r = postprocess(_base_analysis(object_part="keyboard"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["object_part"] == "unknown"

    def test_package_part_on_laptop_row_mapped_to_unknown(self):
        r = postprocess(_base_analysis(object_part="seal"),
                        claim_object="laptop", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["object_part"] == "unknown"


# ---- Rule 3: risk_flags split / validate / rejoin ----

class TestRiskFlagsHandling:
    def test_none_stays_none(self):
        r = postprocess(_base_analysis(risk_flags="none"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["risk_flags"] == "none"

    def test_drops_invalid_tokens_keeps_valid(self):
        r = postprocess(_base_analysis(risk_flags="blurry_image;garbage_xyz;wrong_angle"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        flags = r["risk_flags"].split(";")
        assert "blurry_image" in flags
        assert "wrong_angle" in flags
        assert "garbage_xyz" not in flags

    def test_all_invalid_collapses_to_none(self):
        r = postprocess(_base_analysis(risk_flags="foo;bar"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["risk_flags"] == "none"


# ---- Rule 4: history flag injection (postprocessor, not model) ----

class TestHistoryFlagInjection:
    def test_history_risk_true_injects_flag(self):
        r = postprocess(_base_analysis(risk_flags="none"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=True)
        assert "user_history_risk" in r["risk_flags"].split(";")

    def test_history_risk_false_does_not_inject(self):
        r = postprocess(_base_analysis(risk_flags="none"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert "user_history_risk" not in r["risk_flags"].split(";")

    def test_history_risk_does_not_duplicate(self):
        r = postprocess(_base_analysis(risk_flags="user_history_risk"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=True)
        assert r["risk_flags"].split(";").count("user_history_risk") == 1


# ---- Rule 5 + invariants: consistency enforcement (data-grounded) ----

class TestConsistencyInvariants:
    def test_supported_requires_valid_image_and_evidence(self):
        # supported + invalid image -> must be downgraded away from supported
        r = postprocess(_base_analysis(claim_status="supported", valid_image=False),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["claim_status"] != "supported"

    def test_supported_requires_evidence_met(self):
        r = postprocess(_base_analysis(claim_status="supported", evidence_standard_met=False),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["claim_status"] != "supported"

    def test_valid_image_false_does_NOT_force_evidence_false(self):
        # The data-grounded refinement: user_008 pattern.
        r = postprocess(_base_analysis(
            claim_status="contradicted", valid_image=False,
            evidence_standard_met=True, issue_type="broken_part",
            supporting_image_ids="img_1", severity="high",
        ), claim_object="car", image_paths="a/img_1.jpg", history_risk=True)
        assert r["evidence_standard_met"] is True
        assert r["claim_status"] == "contradicted"


# ---- Rule 6: supporting_image_ids filtering ----

class TestSupportingImageIds:
    def test_filters_to_submitted_ids(self):
        r = postprocess(_base_analysis(supporting_image_ids="img_1;img_99"),
                        claim_object="car",
                        image_paths="a/img_1.jpg;b/img_2.jpg",
                        history_risk=False)
        ids = r["supporting_image_ids"].split(";")
        assert "img_1" in ids
        assert "img_99" not in ids

    def test_empty_collapses_to_none(self):
        r = postprocess(_base_analysis(supporting_image_ids="img_99"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["supporting_image_ids"] == "none"

    def test_supported_row_needs_at_least_one_supporting_id(self):
        # If a supported row has no valid supporting id, it can't be supported.
        r = postprocess(_base_analysis(claim_status="supported", supporting_image_ids="img_99"),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["claim_status"] != "supported"


# ---- Rule 7: no nulls / NaN / empty ----

class TestNoNulls:
    def test_none_reason_filled_with_sentinel(self):
        r = postprocess(_base_analysis(evidence_standard_met_reason=None),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert isinstance(r["evidence_standard_met_reason"], str)
        assert r["evidence_standard_met_reason"] != ""

    def test_empty_justification_filled(self):
        r = postprocess(_base_analysis(claim_status_justification=""),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert isinstance(r["claim_status_justification"], str)
        assert r["claim_status_justification"] != ""

    def test_none_issue_type_filled_with_unknown(self):
        r = postprocess(_base_analysis(issue_type=None),
                        claim_object="car", image_paths="a/img_1.jpg",
                        history_risk=False)
        assert r["issue_type"] in ("unknown", "none")


# ---- End-to-end over a messy model output ----

class TestEndToEndMessy:
    def test_messy_output_becomes_contract_compliant(self):
        messy = {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "  Clear enough. ",
            "risk_flags": "Blurry_Image;;nonsense",
            "issue_type": "Dent",
            "object_part": "Front Bumper",
            "claim_status": "Supported",
            "claim_status_justification": "img_1 shows damage",
            "supporting_image_ids": "img_1;img_2",
            "valid_image": True,
            "severity": "Medium",
        }
        r = postprocess(messy, claim_object="car",
                        image_paths="case_001/img_1.jpg",
                        history_risk=True)
        assert r["issue_type"] == "dent"
        assert r["object_part"] == "front_bumper"
        assert r["claim_status"] == "supported"
        assert r["severity"] == "medium"
        assert "blurry_image" in r["risk_flags"].split(";")
        assert "user_history_risk" in r["risk_flags"].split(";")
        assert r["supporting_image_ids"] == "img_1"   # img_2 not submitted
