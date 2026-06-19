from typing import List, Dict, Any

def normalize_value(value: Any) -> str:
    """
    Normalizes a value for comparison.
    Converts to string, lowercases, and strips whitespace.
    Booleans are converted to their lowercase string representation.
    None is converted to 'none'.
    """
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip().lower()

def calculate_accuracy(expected: List[str], predicted: List[str]) -> float:
    """
    Calculates exact-match accuracy between two lists of equal length.
    Returns 0.0 if the lists are empty.
    Raises ValueError if lists are of different lengths.
    """
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted lists must be of the same length.")

    if not expected:
        return 0.0

    matches = sum(1 for e, p in zip(expected, predicted) if normalize_value(e) == normalize_value(p))
    return matches / len(expected)

def select_winning_strategy(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Selects the winning strategy based on the mean accuracy.
    In case of a tie, prefers Strategy B (two-pass) for determinism.
    metrics should be like: {"strategy_a": {"mean": 0.8}, "strategy_b": {"mean": 0.9}}
    """
    mean_a = metrics.get("strategy_a", {}).get("mean", 0.0)
    mean_b = metrics.get("strategy_b", {}).get("mean", 0.0)

    if mean_a > mean_b:
        return "strategy_a"
    else:
        return "strategy_b"
