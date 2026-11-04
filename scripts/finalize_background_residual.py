"""Independent completed-state audit and actual-GT report for I10."""
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from variants.staged_reconstruction.prepare_data import ROOT, read, write, sha
from variants.staged_reconstruction.background_residual.prepare import EXP, WORK, PARENT, PARENT_SHA, verify

OUT = ROOT/'output/staged_reconstruction/background_residual/screen'


def equal_tree(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def finite_tree(a):
    if torch.is_tensor(a):
        return bool(a.isfinite().all())
    if isinstance(a, dict):
        return all(finite_tree(v) for v in a.values())
    if isinstance(a, (tuple, list)):
        return all(finite_tree(v) for v in a)
    if isinstance(a, float):
        return np.isfinite(a)
    return True


def audit(arm, plan, parent):
    directory = WORK/'formal'/arm
    initial = torch.load(directory/'initial.pt', map_location='cpu', weights_only=False)
    path = directory/'checkpoint.pt'; assert sha(path) == read(directory/'checkpoint.json')['sha256']
    current = torch.load(path, map_location='cpu', weights_only=False)
    assert read(directory/'terminal.json')['status'] == 'completed'
    assert current['iteration'] == plan['actual_updates']
    assert current['phase'] == arm and current['parent_sha256'] == PARENT_SHA
    assert current['protocol_sha256'] == sha(EXP/'protocol.json')
    for field in ['model', 'optimizer', 'camera_stack', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda', 'physics_graphs']:
        assert equal_tree(initial[field], parent[field]), field
    assert finite_tree(current['model']) and finite_tree(current['optimizer'])
    history = [json.loads(v) for v in (directory/'steps.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['local_step'] for r in history] == list(range(1, plan['actual_updates']+1))
    assert finite_tree(history)
    rng = random.Random(); rng.setstate(initial['rng_python']); stack = list(initial['camera_stack'])
    physics_updates = 0
    for row in history:
        if not stack:
            stack = rng.sample(range(120), 120)
        assert stack.pop() == row['frame_index']
        assert row['absolute_step'] == 6000+row['local_step']
        assert row['neutralization_enabled'] == (arm == 'I10_static_residual')
        assert ('physics' in row) == (row['absolute_step'] % 10 == 0)
        physics_updates += int('physics' in row)
    assert current['rng_python'] == rng.getstate() and current['camera_stack'] == stack
    for field in ['rng_numpy', 'rng_torch', 'rng_cuda']:
        assert equal_tree(initial[field], current[field])
    physics_ids = {i for group in current['optimizer']['param_groups'] if group['name'] == 'PIDG_velocity_stress' for i in group['params']}
    assert current['optimizer']['param_groups'] == initial['optimizer']['param_groups']
    for index, values in current['optimizer']['state'].items():
        expected = physics_updates if index in physics_ids else plan['actual_updates']
        assert float(values['step']-initial['optimizer']['state'][index]['step']) == expected
    for name, value in initial['model'].items():
        if name.startswith('static.'):
            assert torch.equal(value, current['model'][name])
    for graph in current['physics_graphs']:
        if graph['neighbors'] is not None:
            assert (graph['neighbors'] >= 0).all() and (graph['neighbors'] < 15507).all()
    report = {'arm': arm, 'checkpoint_sha256': sha(path), 'initial_sha256': sha(directory/'initial.pt'),
        'actual_optimizer_updates': plan['actual_updates'], 'actual_original_physics_updates': physics_updates,
        'initial_model_Adam_all_RNG_sampler_physics_graphs_exact_parent': True,
        'sampler_all_RNG_Adam_counters_checked': True, 'finite_full_state': True, 'static_tensors_unchanged': True,
        'history_sha256': sha(directory/'steps.jsonl'), 'graph_last_values': [g['last'] for g in current['physics_graphs']]}
    return report, current, [r['frame_index'] for r in history]


def comparison():
    receipt = read(OUT/'receipt.json'); names = list(receipt['models'])
    probes = [(188, 43), (188, 146), (86, 86)]
    width = 480; panels = []; font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    sources = []
    for camera, frame in probes:
        values = []
        for name in names:
            directory = OUT/name/f'fixed{camera}'
            paths = [directory/f't{frame:04d}_native.png', directory/f't{frame:04d}_alpha.png']
            rgb, alpha = [np.asarray(Image.open(p)) for p in paths]
            values.append((rgb, alpha))
            sources += [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths]
        union = np.maximum.reduce([a for _, a in values]) > 12
        ys, xs = np.nonzero(union)
        a, b, c, d = (max(0, xs.min()-20), max(0, ys.min()-20), min(1600, xs.max()+21), min(900, ys.max()+21)) if len(xs) else (0, 0, 1600, 900)
        height = round((d-b)*width/(c-a))
        for kind in [0, 1]:
            strip = Image.new('RGB', (width*len(names), height+26), '#171a22'); draw = ImageDraw.Draw(strip)
            for column, (name, value) in enumerate(zip(names, values)):
                im = Image.fromarray(value[kind][b:d, a:c]).convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
                strip.paste(im, (column*width, 26))
                draw.text((column*width+4, 5), f'{name}; c{camera}/t{frame}; '+('RGB' if kind == 0 else 'dynamic alpha'), fill='white', font=font)
            panels.append(strip)
    destination = OUT/'key_probe_comparison.png'
    canvas = Image.new('RGB', (width*len(names), sum(p.height for p in panels)), '#171a22'); y = 0
    for panel in panels:
        canvas.paste(panel, (0, y)); y += panel.height
    canvas.save(destination)
    write(EXP/'comparison_sources.json', {'figure': destination.relative_to(ROOT).as_posix(), 'sha256': sha(destination),
        'sources': sources, 'actual_visual_review': False, 'common_crop_per_probe': True, 'RGB_or_alpha_repair': False})


def main():
    plan = verify(); parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    assert read(EXP/'preflight.json')['passed']
    audits = []; states = []; schedules = []
    for arm in plan['arms']:
        report, state, schedule = audit(arm, plan, parent)
        audits.append(report); states.append(state); schedules.append(schedule)
    assert schedules[0] == schedules[1]
    for field in ['camera_stack', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda']:
        assert equal_tree(states[0][field], states[1][field])
    write(EXP/'formal_state_audit.json', {'passed': True, 'arms': audits, 'matched_all2000_camera_samples_and_final_RNG': True,
        'parent_sha256': sha(PARENT), 'source_freeze_sha256': sha(EXP/'source_freeze.json'),
        'script_sha256': sha(Path(__file__)), 'test_RGB_used': False})
    receipt = read(OUT/'receipt.json')
    for name, item in receipt['models'].items():
        assert sha(ROOT/item['path']) == item['sha256']
    numeric = {}
    for name, result in receipt['validation'].items():
        m = result['metrics']
        checks = {'dynamic_PSNR': m['dynamic_PSNR'] >= plan['reference_dynamic_PSNR']-plan['acceptance']['validation_dynamic_PSNR_drop_max_dB'],
            'dynamic_alpha': m['dynamic_alpha'] >= plan['acceptance']['validation_dynamic_alpha_min'],
            'static_preservation': plan['static_reference_PSNR']-m['static_PSNR'] <= plan['static_PSNR_drop_max_dB']}
        numeric[name] = {'checks': checks, 'passed': all(checks.values()),
            'dynamic_PSNR_delta_from_S3U': m['dynamic_PSNR']-receipt['validation']['S3U_parent']['metrics']['dynamic_PSNR'],
            'static_PSNR_drop_from_S1': plan['static_reference_PSNR']-m['static_PSNR']}
    write(EXP/'numerical_checks.json', {'models': numeric, 'numerical_checks_alone_do_not_admit_completeness': True})
    comparison()
    lines = ['# I10: neutralize frozen-background RGB residuals', '',
        'Both arms actually completed2000 native1600 updates from exact S3-U final6k,',
        'including200 original-physics steps. Full-state/sampler/RNG audit passes.',
        'The source freeze, baseline tensors, Adam groups and frozen S1 are unchanged.', '',
        '| Model | Actual-GT val dynamic PSNR | Dynamic alpha | Static-region PSNR |',
        '|---|---:|---:|---:|']
    for name, result in receipt['validation'].items():
        m = result['metrics']; lines.append(f'|{name}|{m["dynamic_PSNR"]:.6f}|{m["dynamic_alpha"]:.6f}|{m["static_PSNR"]:.6f}|')
    a, b = [receipt['validation'][arm]['metrics'] for arm in plan['arms']]
    lines += ['', f'Candidate minus matched control dynamic PSNR: {b["dynamic_PSNR"]-a["dynamic_PSNR"]:+.6f}dB.',
        'These metrics use actual validation RGB, never the neutralized training target.',
        'No test RGB is used. Direct foreground/known-empty targets and original physics',
        'were retained; SSIM neighborhoods still couple adjacent image regions.', '',
        'The separate control_repeat_audit.json records a substantial difference',
        'between historical I0 and this control despite exact initial model/Adam/RNG',
        'and camera schedules. Single-pair PSNR differences are not stable causal',
        'estimates; no best repeat is substituted for the preregistered control.', '',
        'This intervention changes continuation-time RGB supervision only. It cannot',
        'rule out static-derived initialization or damage accumulated before final6k,',
        'and does not establish static geometry truth or a causal percentage.', '',
        'Original6x9 fixed-camera native RGB/dynamic RGB/alpha images have been rendered.',
        'Actual visual review is recorded separately in visual_review.json; numerical',
        'checks do not prove completeness. The reviewed key probes retain missing',
        'surfaces/fragments, including dynamic-only renders. The candidate fails',
        'completeness and is not promoted. No6x200 acceptance pass is claimed.', '',
        'Reproduction: the opt-in background_residual prepare/preflight/train/evaluate',
        'modules and scripts.finalize_background_residual. Preparation and evaluation',
        'preserve prior receipts; repeats require a new versioned experiment.', '']
    (EXP/'assessment.md').write_text('\n'.join(lines), encoding='utf-8')
    verify()
    print(json.dumps({'numeric': numeric, 'validation': {k: v['metrics'] for k, v in receipt['validation'].items()}}, indent=2))


if __name__ == '__main__':
    main()
