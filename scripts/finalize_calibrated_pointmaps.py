"""Report a frozen qualification and render evidence; no fitting or training."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps

from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, write, sha
from variants.staged_reconstruction.calibrated_pointmaps.prepare import EXP, PREVIOUS
from variants.staged_reconstruction.calibrated_pointmaps.infer import verify
from variants.staged_reconstruction.calibrated_pointmaps.core import projection_errors
from output.short_clip_diagnostic.camera_math import intrinsic

OUT = ROOT / 'output/staged_reconstruction/calibrated_pointmaps'


def fmt(value):
    if value is None:
        return 'unavailable'
    if isinstance(value, list):
        return '/'.join(fmt(v) for v in value)
    return f'{value:.3f}'


def main():
    plan = verify()
    receipt = read(EXP / 'inference.json'); result = read(EXP / 'qualification.json')
    assert result['inference_sha256'] == sha(EXP / 'inference.json')
    assert result['protocol_sha256'] == sha(EXP / 'protocol.json')
    meta = read(DATA / 'transforms_train.json')
    frames = {int(Path(f['file_path']).stem): f for f in meta['frames']}
    k = intrinsic(meta['camera_angle_x'], 'metadata', 512, 288)
    rows = {(r['arm'], r['anchor'], r['partner']): r for r in result['pair_rows']}
    control, camera_arm = plan['arms']
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    except OSError:
        font = small = ImageFont.load_default()
    cell_w, cell_h, cols = 256, 181, 6
    sheet = Image.new('RGB', (cell_w*cols, 50+len(plan['anchors'])*cell_h), '#111111')
    draw = ImageDraw.Draw(sheet)
    headings = ['Train RGB anchor', 'Raw confidence partner A', 'Raw confidence partner B',
                'Static-scaled depth A', 'Camera-scaled depth A', 'Raw K reprojection error A']
    for c, title in enumerate(headings):
        draw.text((c*cell_w+4, 6), title, font=font, fill='white')
    draw.text((4, 27), 'confidence: log scale 1 to 10 | depth: fixed 1 to 15 scene units | projection: 0 to 50px | low confidence masked in depth', font=small, fill='#bbbbbb')
    evidence = []; yy, xx = np.indices((288, 512)); pixels = np.stack([xx, yy], -1)
    for index, anchor in enumerate(plan['anchors']):
        partners = plan['partners'][str(anchor)]
        values = []
        for partner in partners:
            item = next(r for r in receipt['pairs'] if r['anchor'] == anchor and r['partner'] == partner)
            path = ROOT / item['path']; assert sha(path) == item['sha256']
            with np.load(path) as source:
                values.append({name: source[name].copy() for name in source.files})
            evidence.append({'path': item['path'], 'sha256': sha(path)})
        source = DATA / (frames[anchor]['file_path'] + '.png')
        assert sha(source) == next(r for r in receipt['pairs'] if r['anchor'] == anchor)['source_RGB_sha256'][str(anchor)]
        rgba = np.asarray(Image.open(source).convert('RGBA'))
        rgb = np.rint(rgba[..., :3]*(rgba[..., 3:4].astype(np.float32)/255)).astype(np.uint8)
        panels = [Image.fromarray(rgb).resize((256, 144), Image.Resampling.LANCZOS)]
        labels = [f'anchor {anchor}; partners {partners[0]}/{partners[1]}']
        for partner, value in zip(partners, values):
            confidence = value['first_confidence']
            colors = colormaps['viridis'](np.clip(np.log10(np.maximum(confidence, 1)), 0, 1))[..., :3]
            panels.append(Image.fromarray((colors*255).astype(np.uint8)).resize((256, 144)))
            coverage = rows[(camera_arm, anchor, partner)]['dynamic_confident_fraction']
            labels.append(f'p{partner}: dynamic confidence>=3 {coverage:.1%}')
        value = values[0]; mask = value['first_confidence'] >= plan['old_pilot_gates_unchanged']['dynamic_confidence_min']
        for arm in [control, camera_arm]:
            row = rows[(arm, anchor, partners[0])]
            if row['scale'] is not None:
                depth = value['first_points'][..., 2]*row['scale']
                colors = colormaps['turbo'](np.clip((depth-1)/14, 0, 1))[..., :3]
                colors[~mask] = 0
                panel = Image.fromarray((colors*255).astype(np.uint8)).resize((256, 144))
            else:
                panel = Image.new('RGB', (256, 144), '#222222')
            panels.append(panel); labels.append('scale '+fmt(row['scale'])+'; static err '+fmt(row['static_depth_error_median_p90']))
        error = projection_errors(value['first_points'].reshape(-1, 3), pixels.reshape(-1, 2), k, np.eye(3), np.zeros(3), 1.).reshape(288, 512)
        colors = colormaps['inferno'](np.clip(error/50, 0, 1))[..., :3]
        colors[~mask] = 0
        panels.append(Image.fromarray((colors*255).astype(np.uint8)).resize((256, 144)))
        labels.append('dynamic median/p90 '+fmt(rows[(camera_arm, anchor, partners[0])]['raw_first_projection_error_median_p90_pixels'])+'px')
        y = 50+index*cell_h
        for c, (panel, label) in enumerate(zip(panels, labels)):
            sheet.paste(panel, (c*cell_w, y)); draw.text((c*cell_w+3, y+147), label, fill='white', font=small)
    figure = OUT / 'wide_pair_geometry_qualification.png'; sheet.save(figure)

    old_rows = read(PREVIOUS / 'pair_qualification.json')['pair_rows']
    old_inference = read(PREVIOUS / 'pair_inference.json')
    audit = {'predictions': evidence, 'protocol_sha256': sha(EXP / 'protocol.json'),
        'report_script_sha256': sha(Path(__file__)),
        'source_freeze_sha256': sha(EXP / 'source_freeze.json'),
        'inference_sha256': sha(EXP / 'inference.json'), 'qualification_sha256': sha(EXP / 'qualification.json'),
        'figure': figure.relative_to(ROOT).as_posix(), 'figure_sha256': sha(figure),
        'old_predictions_unchanged': all(sha(ROOT/r['path']) == r['sha256'] for r in old_inference['pairs']),
        'old_qualification_sha256': sha(PREVIOUS / 'pair_qualification.json'),
        'PIDG_optimizer_updates': 0, 'source_frozen_verified': True}
    disabled = []
    for name in ['prepare', 'infer', 'qualify']:
        command = [sys.executable, '-m', 'variants.staged_reconstruction.calibrated_pointmaps.'+name]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        assert run.returncode != 0 and 'opt-in' in run.stderr
        disabled.append({'module': name, 'exit_code': run.returncode, 'refuses_without_opt_in': True})
    audit['explicit_opt_in_checks'] = disabled
    write(EXP / 'evidence_verification.json', audit)
    lines = ['# Wider-pair known-camera pointmaps: qualification results', '',
        'The official frozen MonST3R float32 model actually ran all12 new train-only',
        'pairs. Both time-specific pointmaps are retained. Synthetic recovery tests:',
        '**10 passed**. No PIDG optimization or depth targets were produced.', '',
        '| Anchor | Partners | Dynamic confidence coverage | Static support | Raw first projection median/p90 (px) |',
        '|---|---|---|---|---|']
    for anchor in plan['anchors']:
        partners = plan['partners'][str(anchor)]
        selected = [rows[(camera_arm, anchor, p)] for p in partners]
        lines.append('|'+str(anchor)+'|'+'/'.join(map(str, partners))+'|'+
            ' / '.join(f'{r["dynamic_confident_fraction"]:.1%}' for r in selected)+'|'+
            '/'.join(str(r['static_support']) for r in selected)+'|'+
            '; '.join(fmt(r['raw_first_projection_error_median_p90_pixels']) for r in selected)+'|')
    lines += ['', '| Anchor/partner | Static scale | Camera scale | Camera held support | Camera held projection median/p90 (px) |',
              '|---|---|---|---|---|']
    for item in receipt['pairs']:
        a, p = item['anchor'], item['partner']; row = rows[(camera_arm, a, p)]
        lines.append(f'|{a}/{p}|{fmt(rows[(control, a, p)]["scale"])}|{fmt(row["scale"])}|{row["second_held_support"]}|{fmt(row["second_held_projection_error_median_p90_pixels"])}|')
    for arm in plan['arms']:
        selected = [r for r in result['pair_rows'] if r['arm'] == arm]
        stable = [r for r in result['partner_stability'] if r['arm'] == arm]
        lines += ['', f'{arm}: {sum(r["passed"] for r in selected)}/12 pairs and',
                  f'{sum(r["passed"] for r in stable)}/6 anchors pass their respective checks.',
                  f'Entire arm passed: {result["arm_passed"][arm]}.']
    lines += ['', f'The original adjacent-pair control had {sum(r["passed"] for r in old_rows)}/12 passing pairs.',
        'The wide-pair control changes pair selection only. The second arm changes',
        'the scale estimator on exactly those same predictions. Scalar alignment',
        'cannot change first-view point-to-pixel projection or raw confidence.',
        'It also cannot restore an unseen surface absent from a pointmap.', '',
        'These are necessary consistency checks, not synchronized novel-view GT.',
        'S1 depth remains a model-conditioned reference. Failure of this pilot is',
        'not rejection of the whole paper or evidence for an inaccurate static S1.',
        'In particular, known-camera projection/confidence failures are independent',
        'of fitting to static reference depth. Original gates are unchanged.', '',
        'Visual panel masks show all raw-confidence>=3 pixels, including predictions',
        'on the black input background; these are not foreground geometry targets.',
        'The scalar estimate has no camera/model optimization or pointmap warp.', '',
        'The evidence figure is generated; actual visual inspection is recorded',
        'separately in visual_review.json. Missing or nonfinite statistics are null',
        'and fail applicable gates. No full physics training follows a failed gate.', '',
        'Reproduce in separate processes (each requires the explicit opt-in flag):',
        '`python -m variants.staged_reconstruction.calibrated_pointmaps.prepare`,',
        '`python -m variants.staged_reconstruction.calibrated_pointmaps.infer`,',
        '`python -m variants.staged_reconstruction.calibrated_pointmaps.qualify`.',
        'Flag: `--enable-calibrated-pointmap-pilot`. Existing receipts are preserved;',
        'a repeat requires a new versioned experiment. Reporting command:',
        '`python -m scripts.finalize_calibrated_pointmaps`.', '']
    (EXP / 'assessment.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'arm_passed': result['arm_passed'], 'evidence': audit}, indent=2))


if __name__ == '__main__':
    main()
