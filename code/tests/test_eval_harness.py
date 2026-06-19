"""Sprint 5.8 audit (W7) — tests for evaluation/main.py + report.py.

These cover the winner-selection logic (previously inline and untested) and
the report generator's rendering of the measured / design-basis cases. Pure,
no network. Run from code/:  pytest tests/test_eval_harness.py -v
"""
import json
from pathlib import Path

import pytest

from evaluation.main import select_winner
from evaluation import report


# ---------------- select_winner ----------------

class TestSelectWinner:
    def _res(self, single_acc, two_acc):
        return {
            "single": {"mean_accuracy": single_acc},
            "two": {"mean_accuracy": two_acc},
        }

    def test_higher_accuracy_wins(self):
        assert select_winner(self._res(0.8, 0.6)) == "single"
        assert select_winner(self._res(0.6, 0.8)) == "two"

    def test_tie_goes_to_two(self):
        assert select_winner(self._res(0.7, 0.7)) == "two"

    def test_single_strategy_present(self):
        # if only one strategy has results, that one wins
        assert select_winner({"single": {"mean_accuracy": 0.5}}) == "single"
        assert select_winner({"two": {"mean_accuracy": 0.5}}) == "two"

    def test_missing_accuracy_treated_as_zero(self):
        # robust to a malformed entry missing mean_accuracy
        assert select_winner({"single": {}, "two": {"mean_accuracy": 0.6}}) == "two"

    def test_empty_defaults_to_two(self):
        assert select_winner({}) == "two"


# ---------------- report rendering ----------------

class TestReportRendering:
    def _summary_measured(self):
        return {
            "winner": "single", "selection_basis": "measured", "limit": None,
            "single": {"n_rows": 20, "mean_accuracy": 0.7,
                       "overall_exact_match": 0.3,
                       "claim_status_prf": {"macro": {"f1": 0.78, "precision": 0.8,
                                                      "recall": 0.76, "support": -1}},
                       "per_field_accuracy": {"claim_status": 0.85, "issue_type": 0.45,
                                              "object_part": 0.8, "severity": 0.45,
                                              "valid_image": 0.85, "evidence_standard_met": 0.8},
                       "field_error_counts": {"claim_status": 3, "issue_type": 11},
                       "confusion_matrix": {"supported": {"supported": 12, "contradicted": 0,
                                                          "not_enough_information": 0},
                                            "contradicted": {"supported": 1, "contradicted": 3,
                                                             "not_enough_information": 1},
                                            "not_enough_information": {"supported": 1, "contradicted": 0,
                                                                       "not_enough_information": 2}},
                       "stats": {"pass1_calls": 0, "pass2_calls": 20, "cache_hits": 0,
                                 "pass1_failures": 0, "pass2_failures": 0,
                                 "input_tokens": 2666, "output_tokens": 327},
                       "runtime_seconds": 48.5, "avg_latency_seconds": 2.4},
            "two": {"measured": True, "n_rows": 20, "mean_accuracy": 0.65,
                    "overall_exact_match": 0.25,
                    "claim_status_prf": {"macro": {"f1": 0.7}},
                    "per_field_accuracy": {"claim_status": 0.8, "issue_type": 0.4,
                                           "object_part": 0.75, "severity": 0.4,
                                           "valid_image": 0.8, "evidence_standard_met": 0.75},
                    "field_error_counts": {"claim_status": 4, "issue_type": 12},
                    "confusion_matrix": {}, "claim_status_prf_": {},
                    "stats": {"pass1_calls": 20, "pass2_calls": 20, "cache_hits": 0,
                              "pass1_failures": 0, "pass2_failures": 0,
                              "input_tokens": 3888, "output_tokens": 430},
                    "runtime_seconds": 90.0, "avg_latency_seconds": 4.5},
        }

    def test_design_basis_report_states_design_selection(self):
        s = self._summary_measured()
        s["selection_basis"] = "design"
        s["winner"] = "two"
        s["two"] = {"measured": False,
                    "reason": "Two-pass eval did not complete (quota)."}
        text = report._build_report(s)
        assert "selected by design" in text
        assert "could **not** be applied empirically" in text

    def test_design_basis_report_does_not_fabricate_two_pass_numbers(self):
        s = self._summary_measured()
        s["selection_basis"] = "design"
        s["winner"] = "two"
        s["two"] = {"measured": False, "reason": "quota"}
        text = report._build_report(s)
        assert "not measured" in text
        # the fake 0.65 from _summary_measured two-pass must NOT appear
        assert "65.0%" not in text.split("Selection")[0]

    def test_report_includes_output_csv_state_when_present(self):
        s = self._summary_measured()
        s["output_csv_state"] = {"rows": 44, "real_predictions": 30,
                                 "safe_defaults": 14, "note": "backfill pending"}
        text = report._build_report(s)
        assert "Real predictions:** 30 / 44" in text
        assert "Safe-defaults" in text

    def test_report_main_function_round_trips_to_file(self, tmp_path):
        s = self._summary_measured()
        s["output_csv_state"] = {"rows": 44, "real_predictions": 30,
                                 "safe_defaults": 14, "note": "x"}
        # point SUMMARY_JSON/REPORT_MD at tmp via monkeypatch
        orig_summary, orig_report = report.SUMMARY_JSON, report.REPORT_MD
        report.SUMMARY_JSON = tmp_path / "summary.json"
        report.REPORT_MD = tmp_path / "report.md"
        try:
            report.SUMMARY_JSON.write_text(json.dumps(s), encoding="utf-8")
            assert report.main([]) == 0
            assert report.REPORT_MD.exists()
            txt = report.REPORT_MD.read_text(encoding="utf-8")
            assert "# Evaluation Report" in txt
        finally:
            report.SUMMARY_JSON, report.REPORT_MD = orig_summary, orig_report

    def test_report_main_returns_1_when_no_summary(self, tmp_path):
        orig_summary, orig_report = report.SUMMARY_JSON, report.REPORT_MD
        report.SUMMARY_JSON = tmp_path / "does_not_exist.json"
        report.REPORT_MD = tmp_path / "r.md"
        try:
            assert report.main([]) == 1
        finally:
            report.SUMMARY_JSON, report.REPORT_MD = orig_summary, orig_report
