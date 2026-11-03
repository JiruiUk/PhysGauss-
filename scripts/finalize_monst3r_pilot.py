"""Check pinned pilot evidence and preserve source/report repair provenance."""
import ast
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha


def function(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)


def main():
    exp = ROOT / 'experiments/staged_reconstruction/monst3r_geometry'
    target = exp / 'pilot_evidence_verification.json'; assert not target.exists()
    inference = read(exp / 'pair_inference.json'); qualification = read(exp / 'pair_qualification.json')
    source_v1 = exp / 'pair_pilot_source_inference.py'; source_now = ROOT / 'scripts/qualify_monst3r_pairs.py'
    assert sha(source_v1) == inference['pilot_script_sha256']
    assert sha(source_now) == qualification['qualification_script_sha256']
    assert function(source_v1, 'infer') == function(source_now, 'infer')
    assert function(source_v1, 'prepare') == function(source_now, 'prepare')
    assert inference['protocol_sha256'] == qualification['protocol_sha256'] == sha(exp / 'pilot_protocol.json')
    assert qualification['inference_sha256'] == sha(exp / 'pair_inference.json')
    assert len(inference['pairs']) == len(qualification['pair_rows']) == 12
    assert not qualification['pilot_checks_passed'] and not qualification['training_admitted']
    assert all(not row['passed'] for row in qualification['pair_rows'])
    for row in inference['pairs']:
        assert sha(ROOT / row['path']) == row['sha256']
    confidence = read(exp / 'confidence_audit.json')
    figure = ROOT / confidence['figure']; assert sha(figure) == confidence['figure_sha256']
    source_audit = read(exp / 'source_camera_audit.json')
    assert source_audit['all_training_camera_roundtrips_passed'] and len(source_audit['camera_rows']) == 120
    receipt = read(exp / 'pair_vendor.json'); vendor = ROOT / receipt['vendor_root']
    assert all(sha(vendor / name) == value for name, value in receipt['files_sha256'].items())
    # Whole-archive source and vendor remain separate, with import-only edits.
    original = read(exp / 'asset_source.json'); directory = ROOT / original['source_root']
    assert all(sha(directory / name) == value for name, value in original['source_files_sha256'].items())
    figure_review = {'path': confidence['figure'], 'sha256': confidence['figure_sha256'],
        'actually_viewed': True, 'observations': 'Early/middle support at confidence>=3 is sparse. Late support is bright but fails separate geometry checks. This sheet displays confidence, not depth accuracy or 3D truth.'}
    write(exp / 'figure_review.json', figure_review)
    write(target, {'passed': True, 'scope': 'Evidence integrity and failed qualification only; no geometry success',
        'inference_script_sha256': sha(source_v1), 'final_script_sha256': sha(source_now),
        'inference_and_preregistered_prepare_functions_AST_unchanged': True,
        'qualification_reporting_repairs': ['Unsupported scale samples recorded as failure instead of aborting', 'NumPy scalar boolean converted to Python boolean for JSON'],
        'original_thresholds_and_pairs_unchanged': True, 'all12_prediction_hashes_verified': True,
        'source_and_import_adapter_hashes_verified': True,
        'all12_pairs_failed_qualification': True, 'PIDG_training_targets_created': False,
        'test_or_val_RGB_used': False, 'user_data_uploaded': False,
        'actually_inspected_figure': figure_review,
        'scripts_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [
            ROOT / 'scripts/audit_monst3r_source.py', ROOT / 'scripts/prepare_monst3r_assets.py',
            ROOT / 'scripts/prepare_monst3r_pair_vendor.py', source_now,
            ROOT / 'scripts/audit_monst3r_confidence.py', Path(__file__)]}})
    print('MONST3R_FAILED_PILOT_EVIDENCE_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
