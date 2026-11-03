"""Bind legacy lineage, the fixed six RGB observations and new audit source."""
import argparse
from pathlib import Path
from ..prepare_data import ROOT, DATA, read, write, sha

BASE = ROOT/'experiments/staged_reconstruction/S1_joint_revocation_audit'
REPORT = BASE/'v1'
RUNS = ROOT/'runs/pidg_staged_reconstruction/S1_joint_revocation_audit/v1'
OUT = ROOT/'output/staged_reconstruction/S1_joint_revocation_audit/v1'


def stopped():
    return any(p.exists() for p in [ROOT/'stop.request', BASE/'stop.request', RUNS/'stop.request',
        RUNS.parent/'stop.request', ROOT/'runs/pidg_staged_reconstruction/stop.request'])


def verify():
    for name, expected in read(REPORT/'source_freeze.json').items():
        assert sha(ROOT/name) == expected, name


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    assert args.freeze and not stopped() and not (REPORT/'source_freeze.json').exists()
    p = read(BASE/'protocol.json'); prereg = read(BASE/'preregistration.json')
    for name, h in prereg['files'].items():
        assert sha(ROOT/name) == h
    assert sha(ROOT/p['checkpoint']) == p['checkpoint_sha256']
    log = (REPORT/'tests.log').read_text()
    assert 'passed' in log and 'failed' not in log and 'ERROR' not in log
    # Preserve inherited source/data provenance; no old masks are read as a loss.
    frozen = dict(read(ROOT/'experiments/staged_reconstruction/S3_union/source_freeze.json'))
    for name, h in frozen.items():
        assert sha(ROOT/name) == h, name
    paths = list(Path(__file__).parent.glob('*.py')) + [
        BASE/'plan.md', BASE/'protocol.json', BASE/'preregistration.json', BASE/'input_contract.md',
        ROOT/'tests/test_joint_revocation_audit.py', REPORT/'tests.log', ROOT/p['checkpoint'],
        ROOT/'experiments/staged_reconstruction/S3_union/source_freeze.json',
        ROOT/'experiments/staged_reconstruction/S3_union/data_receipt.json',
        ROOT/'experiments/staged_reconstruction/S1_protocol.json', DATA/'transforms_train.json']
    # Bind the complete original native implementation without changing it.
    paths += list((ROOT/'variants/color_repair/code').rglob('*.py'))
    meta = read(DATA/'transforms_train.json'); assert len(meta['frames']) == p['expected_train_views']
    for index in p['fixed_train_list_indices']:
        paths.append(DATA/(meta['frames'][index]['file_path']+'.png'))
    frozen.update({path.relative_to(ROOT).as_posix(): sha(path) for path in paths})
    assert not any('/evaluation_only/' in name for name in frozen)
    write(REPORT/'source_freeze.json', frozen)
    print('J0_FROZEN', len(frozen), sha(REPORT/'source_freeze.json'), flush=True)


if __name__ == '__main__':
    main()
