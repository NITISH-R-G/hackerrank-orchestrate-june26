"""Sprint 2 — Data layer tests (data_loader.py).

Written FIRST (RED) against specs/behavior_spec.md scenarios 6 & 10 and
specs/data_contract.md. Uses the real dataset under dataset/. Run from code/:

    pytest tests/test_data_loader.py -v
"""
import pandas as pd
import pytest

from config import USER_HISTORY_CSV, EVIDENCE_CSV, SAMPLE_CLAIMS_CSV, CLAIMS_CSV
from data_loader import (
    load_all_data,
    get_user_history,
    compute_history_risk,
    get_evidence_requirement,
)


# ---------- load_all_data ----------

class TestLoadAllData:
    def test_load_sample_returns_three_frames(self):
        claims, history, evidence = load_all_data(use_sample=True)
        assert isinstance(claims, pd.DataFrame)
        assert isinstance(history, pd.DataFrame)
        assert isinstance(evidence, pd.DataFrame)
        assert len(claims) == 20          # known sample size
        assert len(history) == 47         # known history size
        assert len(evidence) == 11        # known evidence size

    def test_load_test_claims_when_not_sample(self):
        claims, _, _ = load_all_data(use_sample=False)
        assert len(claims) == 44          # known test size

    def test_claims_have_required_input_columns(self):
        claims, _, _ = load_all_data(use_sample=True)
        for col in ("user_id", "image_paths", "user_claim", "claim_object"):
            assert col in claims.columns

    def test_history_has_risk_columns(self):
        _, history, _ = load_all_data(use_sample=True)
        for col in ("user_id", "past_claim_count", "rejected_claim",
                    "last_90_days_claim_count", "history_flags", "history_summary"):
            assert col in history.columns


# ---------- get_user_history ----------

class TestGetUserHistory:
    def test_returns_dict_for_known_user(self):
        _, history, _ = load_all_data(use_sample=True)
        row = get_user_history(history, "user_001")
        assert row["user_id"] == "user_001"
        assert int(row["past_claim_count"]) == 2

    def test_returns_empty_for_unknown_user(self):
        # Scenario 10: user_id not in history -> empty dict, no crash
        _, history, _ = load_all_data(use_sample=True)
        row = get_user_history(history, "user_does_not_exist")
        assert row == {}


# ---------- compute_history_risk (Scenarios 6 & 10) ----------

class TestComputeHistoryRisk:
    def test_no_history_record_is_not_risky(self):
        # Scenario 10
        assert compute_history_risk({}) is False

    def test_low_ratio_is_not_risky(self):
        # user_001: 0 rejected / 2 total = 0.0, last_90 = 1 -> safe
        assert compute_history_risk({
            "past_claim_count": 2, "rejected_claim": 0,
            "last_90_days_claim_count": 1,
            "history_flags": "none",
            "history_summary": "Low-risk user with prior accepted claims",
        }) is False

    def test_high_rejection_ratio_is_risky(self):
        # user_016: 7 rejected / 11 total = 0.636 > 0.3 -> risky
        assert compute_history_risk({
            "past_claim_count": 11, "rejected_claim": 7,
            "last_90_days_claim_count": 6,
            "history_flags": "user_history_risk;manual_review_required",
            "history_summary": "Frequent rejected car scratch claims",
        }) is True

    def test_high_90_day_volume_is_risky(self):
        # last_90 >= 3 rule
        assert compute_history_risk({
            "past_claim_count": 3, "rejected_claim": 0,
            "last_90_days_claim_count": 3,
            "history_flags": "none",
            "history_summary": "Some claims",
        }) is True

    def test_risk_word_in_flags_is_risky(self):
        assert compute_history_risk({
            "past_claim_count": 1, "rejected_claim": 0,
            "last_90_days_claim_count": 0,
            "history_flags": "manual_review_required",
            "history_summary": "Low risk",
        }) is True

    def test_risk_word_in_summary_is_risky(self):
        assert compute_history_risk({
            "past_claim_count": 1, "rejected_claim": 0,
            "last_90_days_claim_count": 0,
            "history_flags": "none",
            "history_summary": "Prior claim was suspicious",
        }) is True

    def test_division_by_zero_total_does_not_crash(self):
        # past_claim_count = 0 must not raise
        assert compute_history_risk({
            "past_claim_count": 0, "rejected_claim": 0,
            "last_90_days_claim_count": 0,
            "history_flags": "none",
            "history_summary": "New user",
        }) is False

    def test_against_real_dataset_known_risky(self):
        _, history, _ = load_all_data(use_sample=True)
        # user_016 is the canonical high-risk row in the dataset
        row = get_user_history(history, "user_016")
        assert compute_history_risk(row) is True

    def test_against_real_dataset_known_safe(self):
        _, history, _ = load_all_data(use_sample=True)
        row = get_user_history(history, "user_001")
        assert compute_history_risk(row) is False


# ---------- get_evidence_requirement ----------

class TestGetEvidenceRequirement:
    def test_known_car_dent_family(self):
        _, _, evidence = load_all_data(use_sample=True)
        req = get_evidence_requirement(evidence, "car", "dent or scratch")
        assert "panel" in req.lower() or "bumper" in req.lower()

    def test_unknown_family_falls_back(self):
        _, _, evidence = load_all_data(use_sample=True)
        # An unrecognized family must still return *something* (the
        # reviewability / general requirement), never raise.
        req = get_evidence_requirement(evidence, "car", "this is not a family")
        assert isinstance(req, str) and len(req) > 0

    def test_none_family_does_not_crash(self):
        _, _, evidence = load_all_data(use_sample=True)
        req = get_evidence_requirement(evidence, "car", None)
        assert isinstance(req, str) and len(req) > 0
