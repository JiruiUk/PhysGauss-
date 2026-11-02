"""CPU-only independent receipts, sampling, exports and visual review sheets."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from datetime import datetime, timezone
import numpy as np
from PIL import Image, ImageDraw
import torch
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from variants.unified_flow.run import assert_frozen, NAMES, RUNS, REPORTS
from variants.unified_flow.prepare_data import prepare
OUT = Path(__file__).parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4)
assert read(REPORTS/'progress.json')['status'] == 'completed'
assert_frozen(ROOT/'variants/unified_flow/code'); prepare()
prefix_path = RUNS/'shared_prefix/rubber_duck/checkpoints/resume.pt'
prefix_sha = sha(prefix_path)
assert prefix_sha == read(prefix_path.with_suffix('.json'))['sha256']
prefix = torch.load(prefix_path, map_location='cpu', weights_only=False)
configs = [read(RUNS/n/'rubber_duck/metric_config.json') for n in NAMES]
assert {k for k in set(configs[0])|set(configs[1]) if configs[0].get(k) != configs[1].get(k)} == {'stress_source'}
lock = read(RUNS/'selection_lock.json')
lock_time = datetime.fromisoformat(lock['locked_utc']).timestamp()
frames = read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')['frames']
frame_ids = [int(Path(f['file_path']).stem) for f in frames]
assert len(frame_ids) == 50 and frame_ids[21] == 85
checks = {}; schedules = []; moment_schedules = []; sheet = Image.new('RGB', (1600, 1000), 'white')
video_sheet = Image.new('RGB', (1600, 1000), 'white')
for arm_index, name in enumerate(NAMES):
    run = RUNS/name/'rubber_duck'; branch = read(run/'branch.json')
    assert branch['identical_full_checkpoint_sha256'] == prefix_sha
    checkpoint = run/'checkpoints/resume.pt'; manifest = read(checkpoint.with_suffix('.json'))
    assert manifest['iteration'] == 50000 and sha(checkpoint) == manifest['sha256']
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert all(torch.isfinite(v).all() for group in ['deform','grid4d','velocity_net','gaussian_parameters'] for v in state[group].values())
    frozen = [k for k in prefix['velocity_net'] if k.startswith(('time_embed.', 'time_proj.', 'velocity_head.'))]
    if arm_index == 1: frozen = list(prefix['velocity_net'])
    assert all(torch.equal(state['velocity_net'][k], prefix['velocity_net'][k]) for k in frozen)
    extra = ['shared_stress.weight','shared_stress.bias']
    new_changed = any(not torch.equal(state['deform'][k], prefix['deform'][k]) for k in extra)
    old_changed = not torch.equal(state['velocity_net']['stress_head.weight'], prefix['velocity_net']['stress_head.weight'])
    assert new_changed == (arm_index == 1) and old_changed == (arm_index == 0)
    history = [json.loads(s) for s in (run/'metric_history.jsonl').read_text().splitlines()]
    selected = lock['models'][name]
    expected = max(history, key=lambda r: r['metrics']['PSNR'])
    assert expected['iteration'] == selected['iteration'] and abs(expected['metrics']['PSNR']-selected['value']) < 1e-10
    assert (run/'test/ours_best/renders/00000.png').stat().st_mtime > lock_time
    assert len(history) == 51 and len({r['iteration'] for r in history}) == 51
    flow = [json.loads(s) for s in (run/'observed_flow.jsonl').read_text().splitlines()]
    momentum = [json.loads(s) for s in (run/'material_momentum.jsonl').read_text().splitlines()]
    assert len(flow) == 468 and len(momentum) == 420
    pairs = {(p['source'],p['target']) for p in read(REPORTS/'flow_data_manifest.json')['pairs']}
    assert all((r['source'],r['target']) in pairs and abs(r['dt']-(r['target']-r['source'])/199)<1e-14 for r in flow)
    assert len({r['iteration'] for r in flow}) == 468 and all(0<r['valid_fraction']<=1 for r in momentum)
    schedules.append([(r['iteration'],r['source'],r['target']) for r in flow])
    moment_schedules.append([r['iteration'] for r in momentum])
    out = OUT/name; per = read(run/'per_view.json'); max_diff = 0.; videos = []
    for tag in ['best','50000']:
        raw = run/'test'/('ours_'+tag)
        assert len(list((out/'pairs'/tag).glob('*.png'))) == 50
        for i in range(50):
            file = f'{i:05d}.png'
            with Image.open(raw/'gt'/file) as a, Image.open(raw/'renders'/file) as b, Image.open(out/'pairs'/tag/file) as c:
                gt, rendered, joined = [np.asarray(im.convert('RGB')) for im in [a,b,c]]
            assert joined.shape == (950,3200,3)
            assert np.array_equal(gt,joined[50:950,:1600]) and np.array_equal(rendered,joined[50:950,1600:])
            mse = np.mean(((gt.astype(np.float64)-rendered)/255.)**2)
            value = -10*np.log10(mse)
            error = abs(value-per['ours_'+tag]['PSNR'][file]); max_diff=max(max_diff,error)
        assert max_diff < 1e-4
        file = out/f'GT_Render_{tag}.mp4'
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(file))
        assert count == 50 and abs(seconds-5)<.11
        videos.append({'path': str(file),'frames':count,'seconds':seconds,'sha256':sha(file)})
    archive = OUT/(name+'_GT_Render.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert sum(p.endswith('.png') for p in z.namelist()) == 100
    ids = [0,21,25,49]
    for col, i in enumerate(ids):
        with Image.open(out/'pairs/best'/f'{i:05d}.png') as im:
            sheet.paste(im.resize((800,238)), ((col%2)*800,arm_index*500+(col//2)*250))
    reader = imageio_ffmpeg.read_frames(str(out/'GT_Render_best.mp4'), pix_fmt='rgb24')
    meta = next(reader); w,h=meta['size']; assert (w,h)==(3200,950)
    for i, data in enumerate(reader):
        if i in ids:
            col=ids.index(i); im=Image.frombytes('RGB',(w,h),data)
            video_sheet.paste(im.resize((800,238)),((col%2)*800,arm_index*500+(col//2)*250))
    checks[name]={'full_checkpoint_sha256':manifest['sha256'],'all_parameters_finite':True,
        'best_iteration':selected['iteration'],'best_validation_PSNR':selected['value'],
        'flow_evaluations':len(flow),'flow_applied_update_iterations':sum(r['iteration']<50000 for r in flow),
        'momentum_evaluations':len(momentum),'momentum_applied_update_iterations':sum(r['iteration']<50000 for r in momentum),
        'unused_physics_keys_unchanged':frozen,'new_stress_head_changed':new_changed,'old_stress_head_changed':old_changed,
        'pixel_identity_pairs_verified':100,'independent_PSNR_max_error':max_diff,
        'ZIP_crc_passed':True,'ZIP_sha256':sha(archive),'videos':videos}
    del state
assert schedules[0] == schedules[1] and moment_schedules[0] == moment_schedules[1]
sheet.save(OUT/'png_review.png'); video_sheet.save(OUT/'decoded_video_review.png')
fig, axes = plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
for parent,name,label in [('pidg_flow_comparison','K2_flow_trajectory','K2 historical pipeline'),*[(RUNS.name,n,n) for n in NAMES]]:
    run=ROOT/'runs'/parent/name/'rubber_duck'
    rows=[json.loads(s) for s in (run/'metric_history.jsonl').read_text().splitlines()]
    axes[0].plot([r['iteration'] for r in rows],[r['metrics']['PSNR'] for r in rows],label=label)
axes[0].set(xlabel='Total iterations',ylabel='Validation PSNR (dB)',title='Reconstruction: validation only')
motion=read(REPORTS/'motion_validation.json'); best=[r for r in motion['results'] if r['snapshot']=='best']
for offset,key,label in [(-.24,'camera_only_static_EPE_px','Camera only'),(0,'geometric_EPE_px','Trajectory'),(.24,'velocity_EPE_px','Velocity advection')]:
    axes[1].bar(np.arange(2)+offset,[r['metrics'][key] for r in best],width=.23,label=label)
axes[1].set(xticks=np.arange(2),xticklabels=['L0 independent stress','L1 shared stress'],ylabel='Endpoint error (px)',title='Independent estimated 2D flow',ylim=(0,1.6))
for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.savefig(OUT/'comparison.png',dpi=180); plt.close(fig)
receipt={'status':'passed','completed_utc':datetime.now(timezone.utc).isoformat(),'same_extended_prefix_sha256':prefix_sha,
    'only_arm_config_difference':'stress_source','source_and_data_SHA_verified':True,'identical_flow_frame_schedule':True,
    'identical_momentum_iteration_schedule':True,'selection_locked_before_test_render':True,'models':checks,
    'manual_review_pending':True,'review_indices':[0,21,25,49],'review_original_frame_ids':[frame_ids[i] for i in [0,21,25,49]],
    'analysis_scripts_sha256':{p.name:sha(p) for p in OUT.glob('*.py')}}
(REPORTS/'final_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
