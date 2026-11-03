"""Bind one inference candidate to tests, sources, inherited data and protocol."""
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.nuisance_static.prepare import verify


def main():
    verify(); report = ROOT/'experiments/staged_reconstruction/S1_depth_evidence/refinement_v1'
    assert not (report/'source_freeze.json').exists()
    log = report/'tests.log'; assert '18 passed' in log.read_text()
    sources = list((ROOT/'variants/staged_reconstruction/depth_refine').glob('*.py'))
    sources += [ROOT/'tests/test_depth_refine.py',Path(__file__),report.parent/'audit.py',
        report.parent/'refinement_protocol.json',report.parent/'results.json',
        ROOT/'experiments/staged_reconstruction/S1_nuisance/v1/source_freeze.json',log]
    manifest = {p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    manifest.update(read(report.parent/'results.json')['raw_depth_and_real_input_sha256'])
    for case in ['diffuse','glossy']:
        p = ROOT/f'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2/{case}_inference.json'
        manifest[p.relative_to(ROOT).as_posix()] = sha(p)
    write(report/'source_freeze.json',manifest)
    write(report/'tests.json',{'passed':True,'tests':18,'new_tests':8,'inherited_tests':10,
        'disabled_returns_original_object':True,'bounded_vertex_and_disjoint_held_cost_refusal':True,
        'zero_and_nonfinite_updates_rejected':True,'log_sha256':sha(log)})
    print('DEPTH_REFINEMENT_FROZEN',sha(report/'source_freeze.json'))


if __name__ == '__main__': main()
