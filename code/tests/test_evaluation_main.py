import pytest
from evaluation.main import normalize_value, calculate_accuracy, select_winning_strategy

def test_normalize_value():
    assert normalize_value("  Hello  ") == "hello"
    assert normalize_value("HELLO") == "hello"
    assert normalize_value("hello") == "hello"
    assert normalize_value(True) == "true"
    assert normalize_value(False) == "false"
    assert normalize_value(None) == "none"

def test_calculate_accuracy():
    # Exactly matching lists
    expected = ["supported", "dent", "door", "true", "none"]
    predicted = ["supported", "dent", "door", "true", "none"]
    assert calculate_accuracy(expected, predicted) == 1.0

    # Partial match
    predicted_partial = ["supported", "scratch", "door", "false", "none"]
    assert calculate_accuracy(expected, predicted_partial) == 3 / 5

    # Completely wrong
    predicted_wrong = ["contradicted", "scratch", "hood", "false", "unknown"]
    assert calculate_accuracy(expected, predicted_wrong) == 0.0

    # Empty lists
    assert calculate_accuracy([], []) == 0.0

    # Different lengths should raise ValueError
    with pytest.raises(ValueError):
        calculate_accuracy(["a", "b"], ["a"])

def test_select_winning_strategy():
    # Strategy A wins
    metrics_a_wins = {"strategy_a": {"mean": 0.9}, "strategy_b": {"mean": 0.8}}
    assert select_winning_strategy(metrics_a_wins) == "strategy_a"

    # Strategy B wins
    metrics_b_wins = {"strategy_a": {"mean": 0.7}, "strategy_b": {"mean": 0.8}}
    assert select_winning_strategy(metrics_b_wins) == "strategy_b"

    # Tie - B should win
    metrics_tie = {"strategy_a": {"mean": 0.85}, "strategy_b": {"mean": 0.85}}
    assert select_winning_strategy(metrics_tie) == "strategy_b"
