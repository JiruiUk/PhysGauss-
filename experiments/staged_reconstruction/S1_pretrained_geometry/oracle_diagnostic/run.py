"""One evaluator-only diagnostic; does not export GT-derived geometry assets."""
import argparse
from pathlib import Path
import os
import msvcrt
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.pretrained_depth.prepare import BASE, REPORT, RUNS, SOURCE, verify, stopped
from variants.staged_reconstruction.pretrained_depth.evaluate import describe, heat
from variants.staged_reconstruction.synthetic_audit.generate import trace
from experiments.staged_reconstruction.S1_pretrained_geometry.oracle_diagnostic.core import oracle_shape, ordinal_pairs, pair_counts

DIAG = BASE / 'oracle_diagnostic'


def freeze():
    verify(); assert not (DIAG/'source_freeze.json').exists()
    assert '20 passed' in (DIAG/'tests.log').read_text(errors='replace')
    paths = list(DIAG.glob('*.py')) + [DIAG/'protocol.json', DIAG/'plan.md', DIAG/'tests.log',
        ROOT/'tests/test_pretrained_depth_diagnostic.py', REPORT/'inference.json', REPORT/'source_freeze.json',
        SOURCE/'dataset.json']
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    write(DIAG/'source_freeze.json', files)
    print('PRIVILEGED_DIAG_FROZEN', sha(DIAG/'source_freeze.json'), flush=True)


def check():
    verify()
    for path, h in read(DIAG/'source_freeze.json').items(): assert sha(ROOT/path) == h, path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--freeze', action='store_true'); parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); assert args.freeze != args.run
    if args.freeze: return freeze()
    check(); assert not stopped(); assert not (DIAG/'evaluation.json').exists()
    lock = (RUNS/'oracle_diagnostic.lock').open('a+b'); lock.seek(0)
    if not lock.read(1): lock.write(b'0'); lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        for path, h in read(SOURCE/'dataset.json')['files'].items(): assert sha(ROOT/path) == h, path
        for path, h in read(REPORT/'inference.json')['outputs'].items(): assert sha(ROOT/path) == h, path
        protocol = read(DIAG/'protocol.json'); results, figures = {}, {}
        output = ROOT/'output/staged_reconstruction/S1_pretrained_geometry/oracle_diagnostic'; output.mkdir(parents=True, exist_ok=True)
        for case in protocol['cases']:
            meta = read(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json')
            pairs = ordinal_pairs(meta['height'], meta['width']); counts, ordinal = Counter(), Counter()
            errors = {key: [] for key in ['oracle_held_visible', 'oracle_held_thin', 'oracle_at_D1', 'D1_same_pixels']}
            rows, panels = [], []
            for i, row in enumerate(meta['frames']):
                assert not stopped()
                with np.load(RUNS/case/f'view_{i:03d}.npz') as data: raw = data['raw_relative']
                gtpath = ROOT/f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt'/(Path(row['image']).stem+'.npz')
                with np.load(gtpath) as gt:
                    target = gt['static_depth']; visible = gt['observed_static_visible'] & (target > 0)
                    thin = gt['thin_static_visible'] & visible
                    rgb = (gt['observed_rgb']*255).round().astype('uint8')
                    observed = trace(np.asarray(row['world_to_camera']), np.asarray(meta['K']), meta['width'], meta['height'], row['time'], case == 'glossy')
                    assert np.array_equal(observed['rgb'].astype(np.float32), gt['observed_rgb'])
                # These privileged arrays remain in evaluator memory only.
                oracle, held, fitted = oracle_shape(raw, target, visible)
                valid = oracle > 0; selected = held & valid; selected_thin = selected & thin
                error = abs(oracle-target)
                c = {'fitted_views': int(fitted), 'held_visible': int(held.sum()), 'held_valid': int(selected.sum()),
                    'held_good_002': int((selected&(error <= .02)).sum()), 'held_good_005': int((selected&(error <= .05)).sum()),
                    'held_thin': int((held&thin).sum()), 'held_thin_valid': int(selected_thin.sum()),
                    'held_thin_good_002': int((selected_thin&(error <= .02)).sum())}
                counts.update(c); errors['oracle_held_visible'].append(error[selected]); errors['oracle_held_thin'].append(error[selected_thin])
                with np.load(ROOT/f'runs/pidg_staged_reconstruction/S1_depth_refinement/v1/{case}/view_{i:03d}.npz') as d1:
                    baseline = d1['refined']; same = selected & (baseline > 0)
                    errors['oracle_at_D1'].append(error[same]); errors['D1_same_pixels'].append(abs(baseline-target)[same])
                pc = pair_counts(raw, pairs, target, visible, thin, observed['depth'] == 0,
                                 (observed['depth'] > 0) & ~observed['static_visible'])
                ordinal.update(pc); rows.append({'image': row['image'], **c, 'ordinal': pc})
                if i in [0, 24, 72]:
                    width, height = meta['width'], meta['height']
                    panel = Image.new('RGB', (width*3, height+36), '#171923'); draw = ImageDraw.Draw(panel)
                    tiles = [('Input RGB', rgb), ('Privileged shape error 0..0.10', heat(error, selected, .1)),
                             ('D1 error on same pixels', heat(abs(baseline-target), same, .1))]
                    for j, (label, image) in enumerate(tiles):
                        panel.paste(Image.fromarray(image), (width*j, 36)); draw.text((width*j+2, 2), case+' '+Path(row['image']).stem, fill='white')
                        draw.text((width*j+2, 18), label, fill='white')
                    panels.append(np.asarray(panel))
                write(RUNS/'oracle_diagnostic_state.json', {'status': 'running', 'pid': os.getpid(), 'case': case, 'completed': i+1})
            ratio = lambda a, b: ordinal[a]/max(ordinal[b], 1)
            report = {'static_order_error_fraction': ratio('static_ordered_wrong', 'static_ordered'),
                'strong_static_order_error_fraction': ratio('strong_static_ordered_wrong', 'strong_static_ordered'),
                'thin_order_error_fraction': ratio('thin_ordered_wrong', 'thin_ordered'),
                'strong_thin_order_error_fraction': ratio('strong_thin_ordered_wrong', 'strong_thin_ordered'),
                'strong_static_visible_pair_purity': ratio('strong_static_visible', 'strong_pairs'),
                'strong_has_empty_endpoint_fraction': ratio('strong_has_empty_endpoint', 'strong_pairs'),
                'strong_has_dynamic_endpoint_fraction': ratio('strong_has_dynamic_endpoint', 'strong_pairs'),
                'strong_retention_of_separated_static_pairs': ratio('strong_static_ordered', 'static_ordered'),
                'strong_retention_of_separated_thin_pairs': ratio('strong_thin_ordered', 'thin_ordered'),
                'nearly_equal_static_false_strong_fraction': ratio('nearly_equal_static_strong', 'nearly_equal_static')}
            results[case] = {'privileged_counts': dict(counts), 'ordinal_counts': dict(ordinal), 'ordinal_ratios': report,
                'error_diagnostics': {key: describe(np.concatenate(value)) for key, value in errors.items()}, 'frames': rows}
            path = output/f'{case}_privileged_shape.png'; assert not path.exists()
            Image.fromarray(np.concatenate(panels)).save(path); figures[path.relative_to(ROOT).as_posix()] = sha(path)
            print('PRIVILEGED_SHAPE_DIAG', case, results[case]['error_diagnostics'], report, flush=True)
        check()
        write(DIAG/'evaluation.json', {'results': results, 'images': figures, 'GT_privileged_evaluator_only': True,
            'coefficients_metric_arrays_and_masks_not_exported': True, 'not_training_admission': True,
            'original_D3_gate_sha256_unchanged': sha(REPORT/'gate.json'), 'source_freeze_sha256': sha(DIAG/'source_freeze.json')})
        write(RUNS/'oracle_diagnostic_state.json', {'status': 'evaluated_pending_review', 'Gaussian_training_started': False})
    except Exception as error:
        write(RUNS/'oracle_diagnostic_state.json', {'status': 'saved_stopped' if stopped() else 'failed', 'error': str(error)})
        raise
    finally:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1); lock.close()


if __name__ == '__main__': main()
