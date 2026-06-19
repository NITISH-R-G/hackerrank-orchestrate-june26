"""Metrics evaluation module."""
import csv
from pathlib import Path
from typing import Dict, Union


def evaluate_predictions(
    predictions_path: Path, truth_path: Path
) -> Dict[str, Union[int, float]]:
    """Evaluate accuracy of predictions against ground truth."""
    if not predictions_path.exists() or not truth_path.exists():
        return {}

    with open(predictions_path, 'r', encoding='utf-8') as f:
        preds = list(csv.DictReader(f))
    with open(truth_path, 'r', encoding='utf-8') as f:
        truth = {row['user_id']: row for row in csv.DictReader(f)}

    metrics: Dict[str, Union[int, float]] = {
        'total': len(preds),
        'correct_claim_status': 0,
        'correct_issue_type': 0,
        'correct_object_part': 0,
        'correct_severity': 0
    }

    fields = ['claim_status', 'issue_type', 'object_part', 'severity']

    for pred in preds:
        user_id = pred.get('user_id')
        if not user_id or user_id not in truth:
            continue

        t_row = truth[user_id]
        for field in fields:
            if pred.get(field) == t_row.get(field):
                metrics[f'correct_{field}'] += 1  # type: ignore

    total = int(metrics['total'])
    for field in fields:
        correct = int(metrics[f'correct_{field}'])
        acc_key = f'accuracy_{field}'
        metrics[acc_key] = correct / total if total > 0 else 0.0

    return metrics


def main() -> None:
    """Run evaluation on sample claims."""
    base_dir = Path(__file__).parent.parent.parent
    preds_csv = base_dir / "output.csv"
    truth_csv = base_dir / "dataset" / "sample_claims.csv"

    if preds_csv.exists() and truth_csv.exists():
        metrics = evaluate_predictions(preds_csv, truth_csv)
        for key, val in metrics.items():
            print(f"{key}: {val}")


if __name__ == "__main__":
    main()
