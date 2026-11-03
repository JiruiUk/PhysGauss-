"""Retain byte-identical final training logs and small checkpoint receipts."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.scratch_integrity.prefix import PREFIX, REPORT, RUNS
from variants.scratch_integrity.run import NAMES
from variants.metric_tuning.runtime import read_json, write_json, sha256


def main():
    files = {}
    runs = {'common_prefix': PREFIX, **{n: RUNS / 'pilot' / n / 'rubber_duck' for n in NAMES}}
    for name, directory in runs.items():
        receipt = read_json(directory / 'checkpoints/resume.json')
        assert receipt['iteration'] == (15000 if name == 'common_prefix' else 50000)
        assert sha256(directory / 'checkpoints/resume.pt') == receipt['sha256']
        for filename, suffix in [
            ('train.log', 'training.log'), ('metric_history.jsonl', 'metric_history.jsonl'),
            ('metric_config.json', 'metric_config.json'), ('model_config.py', 'model_config.py'),
            ('reproduction.json', 'reproduction.json'), ('branch.json', 'branch.json'),
            ('camera_sampling.jsonl', 'camera_sampling.jsonl'), ('transfer_loss.jsonl', 'transfer_loss.jsonl'),
            ('checkpoints/resume.json', 'final_checkpoint.json'),
            ('checkpoints/resume.metrics.json', 'final_metrics_sidecar.json')]:
            source = directory / filename
            if not source.exists():
                continue
            target = REPORT / f'{name}_{suffix}'
            shutil.copyfile(source, target)
            assert sha256(source) == sha256(target)
            files[target.relative_to(ROOT).as_posix()] = {
                'source': source.relative_to(ROOT).as_posix(), 'sha256': sha256(target)}
    write_json(REPORT / 'training_evidence_files.json', files)
    original = ROOT / 'experiments/transfer_integrity/GT_provenance.json'
    gt = read_json(original)['original_RGBA_sha256']
    assert len(gt) == 200 and all(sha256(ROOT / p) == h for p, h in gt.items())
    write_json(REPORT / 'GT_provenance.json', {
        'original_RGBA_sha256': gt, 'all_200_original_files_verified_unchanged': True,
        'historical_provenance_path': original.relative_to(ROOT).as_posix(),
        'historical_provenance_sha256': sha256(original),
        'split_metadata_sha256': {f'data/derived/kinematic_resplit/transforms_{s}.json':
            sha256(ROOT / f'data/derived/kinematic_resplit/transforms_{s}.json') for s in ['train', 'val', 'test']},
        'novel_camera_time_GT_available': False})
    print('TRAINING EVIDENCE COPIED:', len(files), flush=True)


if __name__ == '__main__':
    main()
