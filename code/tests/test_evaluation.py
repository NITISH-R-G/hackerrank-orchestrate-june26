"""Tests for evaluation module."""
import csv
import pytest
from evaluation.main import evaluate_predictions


def test_evaluate_predictions_missing_files(tmp_path):
    """Test evaluation logic when files are missing."""
    # Pass non-existent paths
    pred_path = tmp_path / "preds.csv"
    truth_path = tmp_path / "truth.csv"

    metrics = evaluate_predictions(pred_path, truth_path)
    assert not metrics


def test_evaluate_predictions_empty_files(tmp_path):
    """Test evaluation logic when files are empty."""
    pred_path = tmp_path / "preds.csv"
    truth_path = tmp_path / "truth.csv"

    # Create empty CSVs with just headers
    headers = [
        "user_id", "claim_status", "issue_type", "object_part", "severity"
    ]
    for path in [pred_path, truth_path]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    metrics = evaluate_predictions(pred_path, truth_path)
    assert metrics["total"] == 0
    assert metrics["accuracy_claim_status"] == pytest.approx(0.0)


def test_evaluate_predictions_accuracy(tmp_path):
    """Test evaluation calculation logic."""
    pred_path = tmp_path / "preds.csv"
    truth_path = tmp_path / "truth.csv"

    headers = [
        "user_id", "claim_status", "issue_type", "object_part", "severity"
    ]

    truth_data = [
        ["user_1", "supported", "dent", "door", "low"],
        ["user_2", "contradicted", "scratch", "hood", "medium"],
        ["user_3", "not_enough_information", "unknown", "unknown", "unknown"],
    ]

    pred_data = [
        # All correct
        ["user_1", "supported", "dent", "door", "low"],
        # 1 correct
        ["user_2", "supported", "scratch", "door", "high"],
        # All correct
        ["user_3", "not_enough_information", "unknown", "unknown", "unknown"],
    ]

    with open(truth_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(truth_data)

    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(pred_data)

    metrics = evaluate_predictions(pred_path, truth_path)

    assert metrics["total"] == 3
    assert metrics["correct_claim_status"] == 2
    assert metrics["correct_issue_type"] == 3
    assert metrics["correct_object_part"] == 2
    assert metrics["correct_severity"] == 2

    assert metrics["accuracy_claim_status"] == pytest.approx(2 / 3)
    assert metrics["accuracy_issue_type"] == pytest.approx(1.0)
    assert metrics["accuracy_object_part"] == pytest.approx(2 / 3)
    assert metrics["accuracy_severity"] == pytest.approx(2 / 3)
