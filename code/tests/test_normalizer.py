"""Sprint 5 — Deterministic normalization layer tests.

Written FIRST (RED). The normalizer maps common free-form descriptions to
the official enums BEFORE fuzzy-matching, so 'back bumper' -> 'rear_bumper'
is deterministic rather than reliant on difflib proximity. Run from code/:

    pytest tests/test_normalizer.py -v
"""
import pytest

from normalizer import (
    normalize_object_part, normalize_issue_type, normalize_severity,
    normalize_claim_status, normalize_risk_flags, NORMALIZER_ALIASES,
)


class TestObjectPartAliases:
    def test_back_bumper_to_rear_bumper(self):
        assert normalize_object_part("back bumper", "car") == "rear_bumper"

    def test_rear_bumper_passthrough(self):
        assert normalize_object_part("rear bumper", "car") == "rear_bumper"

    def test_rear_side_panel_to_quarter_panel(self):
        assert normalize_object_part("rear side panel", "car") == "quarter_panel"

    def test_back_light_to_taillight(self):
        assert normalize_object_part("back light", "car") == "taillight"

    def test_rear_windshield_to_windshield(self):
        assert normalize_object_part("rear windshield", "car") == "windshield"

    def test_mirror_to_side_mirror(self):
        assert normalize_object_part("mirror", "car") == "side_mirror"

    def test_trunk_to_body(self):
        assert normalize_object_part("trunk", "car") == "body"

    def test_already_canonical_kept(self):
        assert normalize_object_part("front_bumper", "car") == "front_bumper"

    def test_laptop_part(self):
        assert normalize_object_part("screen", "laptop") == "screen"

    def test_package_part(self):
        assert normalize_object_part("box", "package") == "box"

    def test_unknown_phrase_returns_unknown(self):
        assert normalize_object_part("flux capacitor", "car") == "unknown"

    def test_wrong_object_part_rejected(self):
        # a laptop part on a car row must not be kept
        assert normalize_object_part("keyboard", "car") == "unknown"


class TestIssueTypeAliases:
    def test_screen_crack_to_crack(self):
        assert normalize_issue_type("screen crack") == "crack"

    def test_water_spill_to_water_damage(self):
        assert normalize_issue_type("water spill") == "water_damage"

    def test_shattered_glass_to_glass_shatter(self):
        assert normalize_issue_type("shattered glass") == "glass_shatter"

    def test_canonical_kept(self):
        assert normalize_issue_type("dent") == "dent"

    def test_unknown_returns_unknown(self):
        assert normalize_issue_type("explosion") == "unknown"


class TestSeverityAliases:
    def test_canonical_kept(self):
        assert normalize_severity("medium") == "medium"

    def test_minor_maps_low(self):
        assert normalize_severity("minor") == "low"

    def test_severe_maps_high(self):
        assert normalize_severity("severe") == "high"

    def test_unknown_returns_unknown(self):
        assert normalize_severity("colossal") == "unknown"


class TestClaimStatus:
    def test_canonical(self):
        assert normalize_claim_status("supported") == "supported"

    def test_typo_fuzzy(self):
        assert normalize_claim_status("supprted") == "supported"

    def test_unknown_falls_back_to_nei(self):
        assert normalize_claim_status("maybe") == "not_enough_information"


class TestRiskFlags:
    def test_split_validate_rejoin(self):
        out = normalize_risk_flags("blurry_image;non_original_image")
        assert set(out.split(";")) == {"blurry_image", "non_original_image"}

    def test_drops_unknown(self):
        out = normalize_risk_flags("blurry_image;frobnicate")
        assert "frobnicate" not in out
        assert "blurry_image" in out

    def test_empty_to_none(self):
        assert normalize_risk_flags("") == "none"
        assert normalize_risk_flags("garbage") == "none"


class TestAliasCoverage:
    def test_alias_map_has_car_entries(self):
        assert "rear bumper" in NORMALIZER_ALIASES["object_part"]
        assert "back bumper" in NORMALIZER_ALIASES["object_part"]
