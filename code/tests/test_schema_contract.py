"""Sprint 1 — Data Contract & Schema tests.

Written FIRST (RED) against specs/data_contract.md, before config.py and
schema.py exist. Run from the code/ directory:

    pytest tests/test_schema_contract.py -v

These tests pin the allowed-value enums, the output column order, and the
three Pydantic models. They must fail until config.py/schema.py are
implemented (GREEN).
"""
import pytest
from schema import ClaimAnalysis, ClaimIntentExtraction, PipelineResult
from config import (
    VALID_CLAIM_STATUS, VALID_ISSUE_TYPES, VALID_SEVERITY,
    VALID_RISK_FLAGS, CAR_PARTS, LAPTOP_PARTS, PACKAGE_PARTS,
    OUTPUT_COLUMNS, PARTS_MAP
)


class TestAllowedValues:
    """Verify the allowed value lists are complete and match the spec."""

    def test_claim_status_values(self):
        assert set(VALID_CLAIM_STATUS) == {"supported", "contradicted", "not_enough_information"}

    def test_issue_types_complete(self):
        expected = {"dent", "scratch", "crack", "glass_shatter", "broken_part",
                    "missing_part", "torn_packaging", "crushed_packaging",
                    "water_damage", "stain", "none", "unknown"}
        assert set(VALID_ISSUE_TYPES) == expected

    def test_severity_values(self):
        assert set(VALID_SEVERITY) == {"none", "low", "medium", "high", "unknown"}

    def test_car_parts_contains_unknown(self):
        assert "unknown" in CAR_PARTS

    def test_laptop_parts_contains_unknown(self):
        assert "unknown" in LAPTOP_PARTS

    def test_package_parts_contains_unknown(self):
        assert "unknown" in PACKAGE_PARTS

    def test_parts_map_keys(self):
        assert set(PARTS_MAP.keys()) == {"car", "laptop", "package"}

    def test_risk_flags_contains_none(self):
        assert "none" in VALID_RISK_FLAGS

    def test_risk_flags_contains_user_history_risk(self):
        assert "user_history_risk" in VALID_RISK_FLAGS

    def test_output_columns_count(self):
        assert len(OUTPUT_COLUMNS) == 14

    def test_output_columns_order(self):
        assert OUTPUT_COLUMNS[0] == "user_id"
        assert OUTPUT_COLUMNS[4] == "evidence_standard_met"
        assert OUTPUT_COLUMNS[9] == "claim_status"
        assert OUTPUT_COLUMNS[13] == "severity"


class TestSchemaInstantiation:
    """Verify Pydantic schemas can be instantiated with valid data."""

    def test_claim_analysis_valid(self):
        obj = ClaimAnalysis(
            evidence_standard_met=True,
            evidence_standard_met_reason="Clear image of damage.",
            risk_flags="none",
            issue_type="dent",
            object_part="door",
            claim_status="supported",
            claim_status_justification="img_1 shows a dent on the door.",
            supporting_image_ids="img_1",
            valid_image=True,
            severity="medium"
        )
        assert obj.claim_status == "supported"

    def test_intent_extraction_valid(self):
        obj = ClaimIntentExtraction(
            claimed_damage_description="Dent on front bumper",
            issue_family="dent or scratch",
            claimed_object_part="front_bumper"
        )
        assert obj.issue_family == "dent or scratch"

    def test_pipeline_result_has_all_output_columns(self):
        result = PipelineResult(
            user_id="u001",
            image_paths="images/test/case_001/img_1.jpg",
            user_claim="My car has a dent",
            claim_object="car",
            evidence_standard_met=True,
            evidence_standard_met_reason="Image is clear.",
            risk_flags="none",
            issue_type="dent",
            object_part="door",
            claim_status="supported",
            claim_status_justification="img_1 shows dent.",
            supporting_image_ids="img_1",
            valid_image=True,
            severity="low"
        )
        result_dict = result.model_dump()
        for col in OUTPUT_COLUMNS:
            assert col in result_dict, f"Missing column: {col}"
