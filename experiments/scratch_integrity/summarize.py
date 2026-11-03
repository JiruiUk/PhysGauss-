"""Collect final measurements without selecting checkpoints or editing metrics."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.metric_tuning.runtime import read_json, write_json
from variants.scratch_integrity.run import NAMES

REPORT = ROOT / 'experiments/scratch_integrity'


def main():
    models = {}
    for name in ['C1_original_50k', *NAMES]:
        models[name] = {split: read_json(REPORT / f'{name}_{split}.json')['metrics'] for split in ['val', 'test']}
    for name in ['common_prefix', *NAMES]:
        models.setdefault(name, {})['constraints'] = read_json(REPORT / f'{name}_constraints.json')['means']
    delta = {}
    for section in ['val', 'test', 'constraints']:
        delta[section] = {key: value - models[NAMES[0]][section][key] for key, value in models[NAMES[1]][section].items()}
    result = {
        'models': models, 'S1_minus_S0': delta,
        'shape': read_json(REPORT / 'shape_summary.json'),
        'support': read_json(REPORT / 'support_summary.json'),
        'causal_comparison': NAMES,
        'historical_C1_is_reference_only': True,
        'training_constraint_diagnostics_are_not_independent_3D_GT': True,
        'novel_view_GT_available': False, 'physical_identity_accuracy_measured': False}
    write_json(REPORT / 'numeric_summary.json', result)
    for name in ['C1_original_50k', *NAMES]:
        print(name, 'val', models[name]['val'], 'test', models[name]['test'])
    print('S1 minus S0', delta)
    for name in ['common_prefix', *NAMES]:
        print(name, 'constraints', models[name]['constraints'])


if __name__ == '__main__':
    main()
