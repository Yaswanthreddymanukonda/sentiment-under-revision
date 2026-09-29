# Compare a fresh run in results/ against the copies we archived.

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

JSON_FILES = [
    'metrics.json',
    'primary_statistics.json',
    'matched_budget.json',
    'data_audit.json',
    'cue_slices.json',
    'error_cases.json',
]


def check_equal(expected, actual, path):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f'{path}: dictionary keys differ')
        for key in expected:
            check_equal(expected[key], actual[key], f'{path}.{key}')
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f'{path}: list lengths differ')
        for i, (left, right) in enumerate(zip(expected, actual)):
            check_equal(left, right, f'{path}[{i}]')
        return

    if isinstance(expected, (int, float)):
        if not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f'{path}: expected {expected}; obtained {actual}')
        return

    if expected != actual:
        raise AssertionError(f'{path}: values differ')


def main():
    for name in JSON_FILES:
        expected = json.loads((ROOT / 'reference_results' / name).read_text())
        actual = json.loads((ROOT / 'results' / name).read_text())
        check_equal(expected, actual, name)

    expected_pred = pd.read_csv(ROOT / 'reference_results' / 'predictions.csv')
    actual_pred = pd.read_csv(ROOT / 'results' / 'predictions.csv')
    pd.testing.assert_frame_equal(expected_pred, actual_pred, check_exact=True)

    rows = {r['key']: r for r in json.loads((ROOT / 'results' / 'metrics.json').read_text())}
    stats = json.loads((ROOT / 'results' / 'primary_statistics.json').read_text())

    summary = {
        'status': 'PASS',
        'checked_json_files': JSON_FILES,
        'prediction_rows_checked': len(actual_pred),
        'prediction_columns_checked': len(actual_pred.columns),
        'original_pair_accuracy': rows['LR_bigrams_original']['pair_accuracy'],
        'augmented_pair_accuracy': rows['LR_bigrams_augmented']['pair_accuracy'],
        'pair_accuracy_gain': stats['pair_accuracy_difference'],
        'note': 'metrics and test predictions match reference_results/. skipped env/runtime.',
    }
    (ROOT / 'results' / 'verification.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
