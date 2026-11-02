"""Same independent post-selection raw-flow diagnostic for the new arms."""
import importlib.util
from pathlib import Path
import sys
import inspect

ROOT = Path(__file__).resolve().parents[2]


def evaluate():
    sys.path.insert(0, str(ROOT/'variants/unified_flow/code'))
    # Load the extended architecture before importing the read-only diagnostic.
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    assert 'unified_flow' in inspect.getfile(DeformModel)
    p = ROOT/'output/flow_comparison/evaluate_motion_v2.py'
    spec = importlib.util.spec_from_file_location('unified_motion_eval', p)
    motion = importlib.util.module_from_spec(spec); spec.loader.exec_module(motion)
    runs = ROOT/'runs/pidg_unified_flow'
    assert (runs/'selection_lock.json').exists()
    manifest = motion.read(motion.CACHE/'manifest.json')
    assert all(motion.sha(motion.CACHE/k) == v for k, v in manifest['output_sha256'].items())
    rows = [motion.evaluate(runs/name/'rubber_duck', tag, manifest)
            for name in ['L0_corrected_trajectory', 'L1_shared_stress'] for tag in ['best', '50000']]
    motion.REPORTS = ROOT/'experiments/unified_flow'
    result = {'status': 'complete', 'results': rows, 'observations': 'same frozen V2 validation raw-flow cache as K1/K2, never used by training',
        'cache_manifest_sha256': motion.sha(motion.CACHE/'manifest.json'),
        'selection_lock_sha256': motion.sha(runs/'selection_lock.json')}
    import json
    (motion.REPORTS/'motion_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
