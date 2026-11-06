"""Read saved L5 evidence; create report assets only. Never load or train a model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EVIDENCE = BASE / 'v1'
REPORT = HERE / 'technical_report_zh.md'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Include all completed-task inputs cited in this report; no weights read.
    paths = sorted(EVIDENCE.glob('*.json')) + [EVIDENCE/'assessment.md', BASE/'protocol.json']
    paths += [ROOT/f'variants/staged_reconstruction/{part}/{name}' for part, name in [
        ('i10_surface_support', 'core.py'), ('i10_surface_support', 'structure.py'),
        ('i10_surface_support', 'runtime.py'), ('i10_temporal_memory', 'core.py'),
        ('i10_revocable_surface', 'core.py'), ('i10_joint', 'runtime.py')]]
    for folder in ['RubberDuck_I10_joint', 'RubberDuck_I10_surface_graph',
                   'RubberDuck_temporal_memory', 'RubberDuck_visibility_audit', 'RubberDuck_joint_pilot']:
        paths.append(ROOT/f'experiments/staged_reconstruction/{folder}/v1/assessment.md')
    for arm in ['temporal_control', 'surface_support']:
        paths.append(ROOT/f'runs/pidg_staged_reconstruction/RubberDuck_surface_support/v1/formal/{arm}/usage.json')
    paths = sorted(set(paths))
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    final = read(EVIDENCE/'final_review.json')
    assert final['status'] == 'completed_not_promoted' and not final['pending']
    for name, expected in final['artifacts_sha256'].items():
        assert sha(EVIDENCE/name) == expected, name
    ev = read(EVIDENCE/'evaluation.json')
    structural = read(EVIDENCE/'structural_diagnostic.json')['models']
    audit = read(EVIDENCE/'controller_audit.json')['arms']
    bank = read(EVIDENCE/'surface_bank.json')
    usage = read(ROOT/'runs/pidg_staged_reconstruction/RubberDuck_surface_support/v1/formal/surface_support/usage.json')
    ref_frames = [int(f) for p in bank['patches'] for f in p['reference']]
    active_frames = [u['index'] for u in usage if u['surface_patches'] > 0]
    thirds = lambda frames: [sum(f//40 == b for f in frames) for b in range(3)]
    means = ev['validation']['means']
    for model in means:
        for metric in ['PSNR', 'SSIM', 'L1']:
            computed = sum(f[model][metric] for f in ev['validation']['frames'])/30
            assert abs(computed-means[model][metric]) < 1e-12
    derived = {
        'scope': 'posthoc report-only arithmetic of saved records; zero training/model inference',
        'dynamic_rows': 15507,
        'covered_rows': bank['summary']['rows'],
        'covered_row_fraction_not_surface_area': bank['summary']['rows']/15507,
        'patch_start_thirds': bank['summary']['temporal_thirds'],
        'reference_patch_frames': len(ref_frames),
        'reference_frame_min_max': [min(ref_frames), max(ref_frames)],
        'reference_distinct_frames': len(set(ref_frames)),
        'reference_patch_frame_thirds': thirds(ref_frames),
        'active_structure_sampled_frames': sorted(set(active_frames)),
        'structure_active_steps_thirds': thirds(active_frames),
        'structure_active_steps': len(active_frames),
        'structure_active_step_fraction': len(active_frames)/2000,
        'candidate_minus_control_PSNR_dB': means['surface_support']['PSNR']-means['temporal_control']['PSNR'],
        'candidate_minus_parent_PSNR_dB': means['surface_support']['PSNR']-means['J6_parent']['PSNR'],
        'connection_penalty_reduction_vs_control_fraction': 1-structural['surface_support']['mean_connection_penalty']/structural['temporal_control']['mean_connection_penalty'],
        'validation_means': means,
        'tracking_diagnostic': ev['training_track_diagnostic'],
        'structural_summary': {m: {k:v for k,v in s.items() if k != 'rows'} for m,s in structural.items()},
        'usage_summary': {m: {k:v for k,v in a.items() if k != 'final_history'} for m,a in audit.items()},
        'high_alpha_proxies': ev['diagnostics']['high_alpha_proxies'],
    }
    assert derived['reference_patch_frame_thirds'] == [354, 6, 0]
    assert derived['structure_active_steps_thirds'] == [158, 8, 0]
    assert derived['structure_active_steps'] == audit['surface_support']['steps_with_structure']
    (HERE/'derived_metrics.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.3), constrained_layout=True)
    models = ['J6_parent', 'temporal_control', 'surface_support']
    labels = ['J6', 'Temporal control', 'Surface candidate']
    colors = ['#67758b', '#d48645', '#167a8b']
    vals = [means[m]['PSNR'] for m in models]
    axs[0,0].bar(labels, vals, color=colors)
    axs[0,0].set_ylim(30.8, 32.2)
    axs[0,0].set_ylabel('Mean PSNR (dB)')
    axs[0,0].set_title('A. All 30 validation RGB images')
    for i,v in enumerate(vals): axs[0,0].text(i, v+.025, f'{v:.3f}', ha='center')
    h = audit['surface_support']['final_history']
    axs[0,1].plot([v['completed_updates'] for v in h], [v['active_memories'] for v in h], 'o-', color=colors[2])
    axs[0,1].set(xlabel='Completed updates at refresh', ylabel='Active memories (count)', ylim=(0,24), title='B. Reliability refresh: candidate')
    axs[0,1].annotate('Last refresh: 1750, not 2000', (1750,12), xytext=(650,5), arrowprops={'arrowstyle':'->'})
    x = np.arange(3)
    axs[1,0].bar(x, derived['reference_patch_frame_thirds'], color=['#167a8b','#d48645','#67758b'])
    axs[1,0].set(xticks=x, xticklabels=['Early (0-39)', 'Middle (40-79)', 'Late (80-119)'], ylabel='Initial patch-frame references (count)', title='C. Support concentrated in early frames', ylim=(0,400))
    for i,v in enumerate(derived['reference_patch_frame_thirds']): axs[1,0].text(i,v+8,str(v),ha='center')
    proxy = {(p['camera'],p['time'],p['model']):p['dynamic_dark_high_alpha'] for p in ev['diagnostics']['high_alpha_proxies']}
    for m,label,color in zip(models, labels, colors):
        axs[1,1].plot([2,100,200], [proxy[(86,t,m)] for t in [2,100,200]], 'o-', label=label, color=color)
    axs[1,1].set(xlabel='Fixed time index (not seconds)', ylabel='Dark / high-alpha pixels (count)', title='D. Camera 86: rendered proxy, not GT')
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Rubber Duck L5: local support gains do not establish global completeness', fontsize=14)
    fig.savefig(HERE/'results_overview.png', dpi=170, facecolor='white')
    plt.close(fig)

    assert all(sha(ROOT/k) == v for k,v in before.items()), 'Evidence changed during report generation'
    report_text = REPORT.read_text(encoding='utf-8')
    targets = re.findall(r'\]\((?:<)?([A-Za-z]:/[^\n]*?)(?:>)?\)', report_text)
    checked = []
    for target in targets:
        clean = re.sub(r':\d+$', '', target)
        # The validation manifest is written after all other checks below.
        if Path(clean) != HERE/'report_validation.json':
            assert Path(clean).exists(), clean
        checked.append(clean)
    image_targets = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report_text)
    from PIL import Image
    for target in image_targets:
        with Image.open(target) as im: im.verify()
    manifest = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'report_only': True, 'training_updates': 0, 'new_native_model_inference': False,
        'completed_evidence_review_manifest_hashes_match': True,
        'all_30_validation_means_recomputed': True,
        'all_input_hashes_unchanged': True,
        'source_sha256': before,
        'checked_absolute_links': sorted(set(checked)),
        'image_files_decode': len(image_targets),
        'generated': {n:sha(HERE/n) for n in ['technical_report_zh.md','derived_metrics.json','results_overview.png','build_report_assets.py']},
        'not_retested': ['native gradients', 'model/checkpoint recovery', 'full video decoding', 'browser playback'],
        'note': 'Those completed checks are cited from frozen L5 records; this report does not rerun experiments.',
    }
    (HERE/'report_validation.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    assert all(Path(p).exists() for p in checked)
    print(json.dumps({'passed': True, 'inputs_unchanged':len(before), 'absolute_links':len(set(checked)), 'images_decode':len(image_targets), 'report':str(REPORT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
