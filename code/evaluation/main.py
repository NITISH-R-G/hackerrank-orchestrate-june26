import csv
import json
from pathlib import Path

def evaluate_predictions(predictions_path: Path, truth_path: Path) -> dict:
    """Evaluate accuracy of predictions against ground truth."""
    if not predictions_path.exists() or not truth_path.exists():
        return {}

    with open(predictions_path, 'r', encoding='utf-8') as f:
        preds = list(csv.DictReader(f))
    with open(truth_path, 'r', encoding='utf-8') as f:
        truth = {row['user_id']: row for row in csv.DictReader(f)}

    metrics = {
        'total': len(preds),
        'correct_claim_status': 0,
        'correct_issue_type': 0,
        'correct_object_part': 0,
        'correct_severity': 0
    }

    for pred in preds:
        user_id = pred.get('user_id')
        if not user_id or user_id not in truth:
            continue

        t = truth[user_id]
        if pred.get('claim_status') == t.get('claim_status'):
            metrics['correct_claim_status'] += 1
        if pred.get('issue_type') == t.get('issue_type'):
            metrics['correct_issue_type'] += 1
        if pred.get('object_part') == t.get('object_part'):
            metrics['correct_object_part'] += 1
        if pred.get('severity') == t.get('severity'):
            metrics['correct_severity'] += 1

    if metrics['total'] > 0:
        metrics['accuracy_claim_status'] = metrics['correct_claim_status'] / metrics['total']
        metrics['accuracy_issue_type'] = metrics['correct_issue_type'] / metrics['total']
        metrics['accuracy_object_part'] = metrics['correct_object_part'] / metrics['total']
        metrics['accuracy_severity'] = metrics['correct_severity'] / metrics['total']
    else:
        metrics['accuracy_claim_status'] = 0.0
        metrics['accuracy_issue_type'] = 0.0
        metrics['accuracy_object_part'] = 0.0
        metrics['accuracy_severity'] = 0.0

    return metrics
