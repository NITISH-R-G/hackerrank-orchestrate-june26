"""Sprint 5 — Evaluation metrics tests.

Written FIRST (RED). metrics.py computes per-field exact-match accuracy,
overall exact-match, precision/recall/F1 for claim_status, a 3x3 confusion
matrix, and field-wise error counts. Run from code/:

    pytest tests/test_metrics.py -v
"""
import pytest

from evaluation.metrics import (
    exact_match_accuracy,
    overall_exact_match,
    claim_status_prf,
    confusion_matrix,
    field_error_counts,
    SCORED_FIELDS,
)


def _pred(status="supported", issue="dent", part="front_bumper",
          severity="medium", valid_image="true", evidence="true"):
    return {
        "claim_status": status, "issue_type": issue, "object_part": part,
        "severity": severity, "valid_image": valid_image,
        "evidence_standard_met": evidence,
    }


GT = [
    _pred("supported", "dent", "front_bumper", "medium"),
    _pred("contradicted", "none", "screen", "none"),
    _pred("not_enough_information", "unknown", "unknown", "unknown",
          valid_image="false", evidence="false"),
]


class TestExactMatch:
    def test_perfect_predictions(self):
        acc = exact_match_accuracy(GT, GT)
        for f in SCORED_FIELDS:
            assert acc[f] == 1.0

    def test_single_field_miss(self):
        pred = [dict(GT[0]), dict(GT[1]), dict(GT[2])]
        pred[0] = dict(pred[0]); pred[0]["issue_type"] = "scratch"
        acc = exact_match_accuracy(GT, pred)
        assert acc["issue_type"] == pytest.approx(2 / 3)
        assert acc["claim_status"] == 1.0

    def test_case_insensitive_stripped(self):
        pred = [
            {"claim_status": "SUPPORTED", "issue_type": "Dent",
             "object_part": "Front_Bumper", "severity": "Medium",
             "valid_image": "True", "evidence_standard_met": "True"},
            dict(GT[1]), dict(GT[2]),
        ]
        acc = exact_match_accuracy(GT, pred)
        assert acc["claim_status"] == 1.0


class TestOverallExactMatch:
    def test_perfect(self):
        assert overall_exact_match(GT, GT) == 1.0

    def test_one_row_one_field_off(self):
        pred = [dict(GT[0]), dict(GT[1]), dict(GT[2])]
        pred[0]["issue_type"] = "scratch"
        assert overall_exact_match(GT, pred) == pytest.approx(2 / 3)


class TestClaimStatusPRF:
    def test_perfect(self):
        prf = claim_status_prf(GT, GT)
        for status in ("supported", "contradicted", "not_enough_information"):
            assert prf[status]["f1"] == 1.0

    def test_macro_present(self):
        prf = claim_status_prf(GT, GT)
        assert "macro" in prf
        assert prf["macro"]["f1"] == 1.0


class TestConfusionMatrix:
    def test_shape_and_labels(self):
        cm = confusion_matrix(GT, GT)
        labels = ["supported", "contradicted", "not_enough_information"]
        assert set(cm.keys()) == set(labels)
        for row in labels:
            assert set(cm[row].keys()) == set(labels)
        # diagonal should hold the counts
        assert cm["supported"]["supported"] == 1
        assert cm["contradicted"]["contradicted"] == 1

    def test_off_diagonal(self):
        pred = [_pred("contradicted"), dict(GT[1]), dict(GT[2])]
        cm = confusion_matrix(GT, pred)
        assert cm["supported"]["contradicted"] == 1


class TestFieldErrorCounts:
    def test_perfect_has_zero_errors(self):
        errs = field_error_counts(GT, GT)
        for f in SCORED_FIELDS:
            assert errs[f] == 0

    def test_counts_errors(self):
        pred = [dict(GT[0]), dict(GT[1]), dict(GT[2])]
        pred[0]["issue_type"] = "scratch"
        pred[1]["severity"] = "high"
        errs = field_error_counts(GT, pred)
        assert errs["issue_type"] == 1
        assert errs["severity"] == 1
        assert errs["claim_status"] == 0


class TestScoredFields:
    def test_scored_fields_match_spec(self):
        assert set(SCORED_FIELDS) == {
            "claim_status", "issue_type", "object_part",
            "severity", "valid_image", "evidence_standard_met",
        }
