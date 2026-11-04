"""Declare graph gates before inspecting the full population graph."""
from ..prepare_data import ROOT, DATA, read, write, sha
from .prepare import EXP, ANCHORS


def main():
    assert not (EXP/'graph_protocol.json').exists(), 'Do not replace a frozen graph protocol'
    previous = ROOT/'experiments/staged_reconstruction/uncertainty_motion'
    native = read(previous/'observed_native120_protocol.json')
    inputs = dict(native['inputs'])
    for path in [ANCHORS, previous/'observed_native120_protocol.json',
                 previous/'anchor_selection/assessment.json', previous/'verification_native120.json']:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in inputs.items():
        assert sha(ROOT/name) == expected, name
    files = list((ROOT/'variants/staged_reconstruction/reliable_motion').glob('*.py'))
    files += [ROOT/'tests/test_reliable_motion.py']
    write(EXP/'graph_protocol.json', {
        'enabled_by_default': False,
        'sources': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        'inputs': inputs,
        'construction': {'radius': .08, 'candidates': 24, 'neighbors': 8,
                         'minimum_sources': 3, 'common_min': 3, 'condition_min': .01},
        'active_target_coverage_min': .8,
        'active_target_definition': 'Frozen parent opacity > .1; including all unsupported active points',
        'reference_policy': 'Highest support among all geometrically eligible train observations',
        'pair_policy': 'At least3 common qualified observations; maximum common-time separation <= .08',
        'source_policy': 'Only independently qualified recurring anchors currently trusted at reference time',
        'rank_policy': 'At least3 noncollinear sources; second/largest covariance eigenvalue >= .01',
        'missing_edges': 'Zero padded weight; do not insert duplicated or unsupported sources',
        'scope': 'Frozen conditional local motion prior; no depth, identity, body or surface GT',
        'test_RGB_read': False, 'optimizer_updates': 0})
    print('RELIABLE_GRAPH_PROTOCOL_FROZEN', sha(EXP/'graph_protocol.json'), flush=True)


if __name__ == '__main__':
    main()
