"""Post-lock evaluation, independent artifact checks and matched C1 contrast."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
from .run import ROOT, RUNS, REPORTS, OUT, NAMES, assert_frozen
from variants.metric_tuning import run as engine


def motion_module():
    os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT/'third_party/PIDG/tmp_build')
    sys.path.insert(0, str(ROOT/'variants/c1_flow_ablation/code'))
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    import inspect
    if 'c1_flow_ablation' not in inspect.getfile(DeformModel):
        raise ValueError('Wrong evaluation architecture')
    path = ROOT/'output/flow_comparison/evaluate_motion_v2.py'
    spec = importlib.util.spec_from_file_location('c1_flow_motion_eval', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def checks(motion):
    from reproduction.scripts.training_checkpoint import load_state, PARAMETERS
    assert_frozen(ROOT/'variants/c1_flow_ablation/code')
    lock = engine.read_json(RUNS/'selection_lock.json')
    assert lock['selection_split'] == 'validation' and set(lock['models']) == set(NAMES)
    configs = [engine.read_json(RUNS/n/'rubber_duck/metric_config.json') for n in NAMES]
    assert {k for k in set(configs[0])|set(configs[1]) if configs[0].get(k) != configs[1].get(k)} == {'observed_flow_enabled'}
    schedules = []; model_proofs = {}
    for name in NAMES:
        directory = RUNS/name/'rubber_duck'; checkpoint = directory/'checkpoints/resume.pt'
        receipt = engine.read_json(checkpoint.with_suffix('.json'))
        assert receipt['iteration'] == 50000 and receipt['sha256'] == engine.sha256(checkpoint)
        full = load_state(checkpoint)
        assert full['iteration'] == 50000
        def finite(v):
            if torch.is_tensor(v): return bool(v.isfinite().all())
            if isinstance(v, dict): return all(finite(x) for x in v.values())
            if isinstance(v, (list,tuple)): return all(finite(x) for x in v)
            return True
        assert all(finite(full[k]) for k in ['gaussian_parameters','velocity_net','grid4d','deform',
            'gaussian_optimizer','deform_optimizer'])
        g, d = motion.load(directory, '50000')
        assert all(torch.equal(v.cpu(), d.grid4d.state_dict()[k].cpu()) for k,v in full['grid4d'].items())
        assert all(torch.equal(v.cpu(), d.deform.state_dict()[k].cpu()) for k,v in full['deform'].items())
        assert all(torch.equal(v.cpu(), g.velocity_net.state_dict()[k].cpu()) for k,v in full['velocity_net'].items())
        assert all(torch.allclose(v.cpu(), getattr(g,k).detach().cpu(), atol=1e-7, rtol=1e-6)
            for k,v in full['gaussian_parameters'].items())
        del g,d,full; torch.cuda.empty_cache()
        branch = engine.read_json(directory/'branch.json')
        assert branch['identical_full_checkpoint_sha256'] == engine.read_json(REPORTS/'source_freeze.json')['full_prefix_sha256']
        sampled = [json.loads(s) for s in (directory/'camera_sampling.jsonl').read_text().splitlines()]
        assert [r['iteration'] for r in sampled] == list(range(3001,50001))
        schedules.append([(r['iteration'],r['frame'],r['time']) for r in sampled])
        best = engine.read_json(directory/'metric_best.json')['PSNR']
        assert best == lock['models'][name]
        assert best['iteration'] <= 50000
        hashes = {str(p.relative_to(directory)):engine.sha256(p) for tag in ['best','50000']
            for folder in ['point_cloud','deform'] for p in (directory/folder/('iteration_'+tag)).rglob('*') if p.is_file()}
        exports = engine.read_json(REPORTS/(name+'_exports.json'))
        assert exports['pairs'] == 100 and exports['pixel_identity_verified']
        assert all(v['frames'] == 50 for v in exports['videos'])
        flow_path = directory/'observed_flow.jsonl'
        events = [json.loads(s) for s in flow_path.read_text().splitlines()] if flow_path.exists() else []
        due = [r for r in sampled if r['iteration'] % 100 == 0]
        if name == NAMES[0]: assert not events
        else:
            assert [r['iteration'] for r in events] == [r['iteration'] for r in due]
            assert all(r['source'] == int(s['frame']) for r,s in zip(events,due))
        model_proofs[name] = {'full_checkpoint_sha256':receipt['sha256'], 'snapshot_sha256':hashes,
            'full_snapshot_parity':True, 'parameters_and_optimizers_finite':True,
            'camera_samples':len(sampled), 'flow_scheduled_events':len(events),
            'flow_evaluations':sum('weighted_loss' in r for r in events),
            'flow_updates':sum('weighted_loss' in r and r['iteration'] < 50000 for r in events),
            'zip_sha256':engine.sha256(OUT/(name+'_GT_Render.zip'))}
    assert schedules[0] == schedules[1]
    return {'status':'passed', 'source_data_SHA_verified':True, 'same_camera_schedule':True,
        'only_config_difference':'observed_flow_enabled', 'models':model_proofs,
        'manual_review_pending':True, 'selection_lock_sha256':engine.sha256(RUNS/'selection_lock.json')}


def review():
    # Early, original85, middle101, late197; original85 remains full-size separately.
    canvas = Image.new('RGB',(1280,800),'#17283b'); draw = ImageDraw.Draw(canvas)
    for col,name in enumerate(NAMES):
        for row,i in enumerate([0,21,25,49]):
            path = OUT/name/'pairs/best'/f'{i:05d}.png'
            with Image.open(path) as im: preview = im.convert('RGB').resize((640,190))
            canvas.paste(preview,(col*640,row*200)); draw.text((col*640+8,row*200+190),f'{name} | index {i}',fill='white')
    canvas.save(OUT/'png_review.png')
    import imageio_ffmpeg
    import subprocess
    decoded = Image.new('RGB',(1280,800),'#17283b')
    for col,name in enumerate(NAMES):
        movie = OUT/name/'GT_Render_best.mp4'
        command = [imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(movie),'-vf','scale=640:190',
            '-f','rawvideo','-pix_fmt','rgb24','-']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw,err = process.communicate()
        assert process.returncode == 0, err.decode(errors='replace')
        assert len(raw) == 50*640*190*3
        frames = np.frombuffer(raw,dtype=np.uint8).reshape(50,190,640,3)
        for row,i in enumerate([0,21,25,49]): decoded.paste(Image.fromarray(frames[i]),(col*640,row*200))
    decoded.save(OUT/'decoded_video_review.png')


def finalize():
    assert (RUNS/'selection_lock.json').exists()
    # The shared exporter independently checks original GT pixels and per-frame PSNR.
    from variants.flow_comparison import export as exporter
    exporter.RUNS, exporter.OUT, exporter.REPORTS = RUNS, OUT, REPORTS
    for name in NAMES:
        for tag in ['best','50000']: (OUT/name/'pairs'/tag).mkdir(parents=True,exist_ok=True)
        exporter.export(RUNS/name/'rubber_duck')
    motion = motion_module(); manifest = motion.read(motion.CACHE/'manifest.json')
    assert all(motion.sha(motion.CACHE/k) == v for k,v in manifest['output_sha256'].items())
    rows = [motion.evaluate(RUNS/n/'rubber_duck',tag,manifest) for n in NAMES for tag in ['best','50000']]
    engine.write_json(REPORTS/'motion_validation.json', {'status':'complete','results':rows,
        'observation_manifest_sha256':motion.sha(motion.CACHE/'manifest.json'),
        'protocol':'fixed observation mask; validation source to next training target; raw UniMatch estimates, not 3D GT'})
    val = {n:engine.read_json(RUNS/n/'rubber_duck/metric_best.json')['PSNR']['metrics'] for n in NAMES}
    test = {n:engine.read_json(RUNS/n/'rubber_duck/results.json') for n in NAMES}
    m = {r['model']:r['metrics'] for r in rows if r['snapshot']=='best'}
    a,b = NAMES
    deltas = {'validation_PSNR_dB':val[b]['PSNR']-val[a]['PSNR'],
        'validation_LPIPS_relative':val[b]['LPIPS']/val[a]['LPIPS']-1,
        'geometry_EPE_relative':m[b]['geometric_EPE_px']/m[a]['geometric_EPE_px']-1,
        'velocity_EPE_relative':m[b]['velocity_EPE_px']/m[a]['velocity_EPE_px']-1}
    criteria = {'image_PSNR':deltas['validation_PSNR_dB'] >= -.2,
        'image_LPIPS':deltas['validation_LPIPS_relative'] <= .05,
        'geometry_motion':deltas['geometry_EPE_relative'] <= .05,
        'velocity_motion':deltas['velocity_EPE_relative'] <= -.10}
    result = {'validation':val,'test':test,'motion':m,'deltas_O1_minus_O0':deltas,
        'prospective_criteria':criteria,'composite_benefit':all(criteria.values()),
        'single_scene_seed_historical_test_exposure':True}
    engine.write_json(REPORTS/'results.json',result)
    engine.write_json(REPORTS/'final_verification.json',checks(motion)); review()
    lines = ['# C1 光流开关完整对照','',
        '两组相同完整3k状态至总50k，仅observed_flow_enabled不同。C1原独立速度/应力、原动量公式、时钟、图像损失和优化器保留。',
        '主比较新O1−新O0；历史C1不是本次无光流控制。二维光流是估计观测，不是物理真值。','',
        '|模型|验证PSNR|验证LPIPS|测试best PSNR|测试best LPIPS|轨迹EPE/px|速度推进EPE/px|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n in NAMES:
        lines.append(f"|{n}|{val[n]['PSNR']:.6f}|{val[n]['LPIPS']:.7f}|{test[n]['ours_best']['PSNR']:.6f}|{test[n]['ours_best']['LPIPS']:.7f}|{m[n]['geometric_EPE_px']:.6f}|{m[n]['velocity_EPE_px']:.6f}|")
    lines += ['', '预先综合门槛通过：'+str(result['composite_benefit']),
        'O1−O0详细变化：`'+json.dumps(deltas)+'`','',
        '只支持这一场景/种子/权重/光流定义的结论；CUDA非确定性限制精确因果解释。未增加外力、本构或轨迹导数速度分支。',
        '人工审阅待完成：查看早中晚、原帧85以及解码视频，补充局部缺失/拖影与运动解释。','',
        f'[同步视频/逐帧浏览]({(OUT/"index.html").as_posix()})']
    (REPORTS/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    cells = ''.join(f'<td><h2>{n}</h2><video id="v{i}" controls loop src="{n}/GT_Render_best.mp4"></video><img id="im{i}"></td>' for i,n in enumerate(NAMES))
    page = '<!doctype html><meta charset="utf-8"><title>C1 光流开关对照</title><style>body{background:#15283c;color:white;font:16px system-ui;margin:24px}table{width:100%}td{width:50%}img,video{width:100%}pre{white-space:pre-wrap}a{color:#9debd8}</style><h1>C1：仅开／关光流</h1><p>左侧O0无光流；右侧O1有光流。每个画面左GT、右Render。</p><button onclick="v0.currentTime=v1.currentTime=0;v0.play();v1.play()">同步播放</button><button onclick="v0.pause();v1.pause()">暂停</button><table><tr>'+cells+'</tr></table><input id="f" type="range" min="0" max="49" value="21"><pre>'+json.dumps(result,ensure_ascii=False,indent=2)+'</pre><script>const names='+json.dumps(NAMES)+';function u(){names.forEach((n,i)=>document.getElementById("im"+i).src=n+"/pairs/best/"+String(f.value).padStart(5,"0")+".png")}f.oninput=u;u()</script>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    print('C1 FLOW COMPARISON EXPORTED AND VERIFIED; MANUAL REVIEW PENDING',flush=True)


if __name__ == '__main__': finalize()
