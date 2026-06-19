"""Evaluation metrics for claim-verification predictions.

Specs: specs/evaluation_spec.md §3-4. Computes per-field exact-match
accuracy, overall exact-match, precision/recall/F1 for claim_status, a 3x3
confusion matrix, and field-wise error counts.

All comparisons are case-insensitive and stripped (booleans compared as
lowercase strings).
"""
from typing import Dict, List

# Fields scored by exact-match (specs/evaluation_spec.md §3).
SCORED_FIELDS = [
    "claim_status", "issue_type", "object_part",
    "severity", "valid_image", "evidence_standard_met",
]

STATUS_LABELS = ["supported", "contradicted", "not_enough_information"]


def _norm(v) -> str:
    return str(v).strip().lower()


def _coerce_bool_str(v) -> str:
    s = _norm(v)
    if s in ("true", "1", "yes"):
        return "true"
    if s in ("false", "0", "no"):
        return "false"
    return s


def _cell(row: dict, field: str) -> str:
    v = row.get(field, "")
    if field in ("valid_image", "evidence_standard_met"):
        return _coerce_bool_str(v)
    return _norm(v)


def exact_match_accuracy(ground_truth: List[dict],
                         predictions: List[dict]) -> Dict[str, float]:
    """Per-field exact-match accuracy in [0,1] for each SCORED_FIELD."""
    n = len(ground_truth)
    correct = {f: 0 for f in SCORED_FIELDS}
    for gt, pred in zip(ground_truth, predictions):
        for f in SCORED_FIELDS:
            if _cell(gt, f) == _cell(pred, f):
                correct[f] += 1
    return {f: (correct[f] / n if n else 0.0) for f in SCORED_FIELDS}


def overall_exact_match(ground_truth: List[dict],
                        predictions: List[dict]) -> float:
    """Fraction of rows where ALL scored fields match."""
    n = len(ground_truth)
    if n == 0:
        return 0.0
    ok = 0
    for gt, pred in zip(ground_truth, predictions):
        if all(_cell(gt, f) == _cell(pred, f) for f in SCORED_FIELDS):
            ok += 1
    return ok / n


def confusion_matrix(ground_truth: List[dict],
                     predictions: List[dict]) -> Dict[str, Dict[str, int]]:
    """3x3 confusion: cm[true][pred] = count over claim_status."""
    cm = {t: {p: 0 for p in STATUS_LABELS} for t in STATUS_LABELS}
    for gt, pred in zip(ground_truth, predictions):
        t = _cell(gt, "claim_status")
        p = _cell(pred, "claim_status")
        if t in cm and p in cm[t]:
            cm[t][p] += 1
    return cm


def claim_status_prf(ground_truth: List[dict],
                     predictions: List[dict]) -> Dict[str, Dict[str, float]]:
    """Per-class precision/recall/F1 + macro average for claim_status."""
    cm = confusion_matrix(ground_truth, predictions)
    prf: Dict[str, Dict[str, float]] = {}
    ps, rs, fs = [], [], []
    for label in STATUS_LABELS:
        tp = cm[label][label]
        fp = sum(cm[other][label] for other in STATUS_LABELS) - tp
        fn = sum(cm[label][other] for other in STATUS_LABELS) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        prf[label] = {"precision": precision, "recall": recall, "f1": f1,
                      "support": tp + fn}
        ps.append(precision); rs.append(recall); fs.append(f1)
    k = len(STATUS_LABELS)
    prf["macro"] = {
        "precision": sum(ps) / k, "recall": sum(rs) / k, "f1": sum(fs) / k,
        "support": -1,
    }
    return prf


def field_error_counts(ground_truth: List[dict],
                       predictions: List[dict]) -> Dict[str, int]:
    """Number of rows where each scored field is wrong."""
    errs = {f: 0 for f in SCORED_FIELDS}
    for gt, pred in zip(ground_truth, predictions):
        for f in SCORED_FIELDS:
            if _cell(gt, f) != _cell(pred, f):
                errs[f] += 1
    return errs


def summarize(ground_truth: List[dict], predictions: List[dict]) -> dict:
    """All metrics in one dict (for the report)."""
    acc = exact_match_accuracy(ground_truth, predictions)
    mean_acc = sum(acc.values()) / len(acc) if acc else 0.0
    return {
        "per_field_accuracy": acc,
        "mean_accuracy": mean_acc,
        "overall_exact_match": overall_exact_match(ground_truth, predictions),
        "claim_status_prf": claim_status_prf(ground_truth, predictions),
        "confusion_matrix": confusion_matrix(ground_truth, predictions),
        "field_error_counts": field_error_counts(ground_truth, predictions),
        "n_rows": len(ground_truth),
    }
