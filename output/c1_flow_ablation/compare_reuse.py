"""Read-only historical control plus newly trained optical-flow candidate."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from output.c1_flow_ablation.reuse_control import CONTROL,TREATMENT,QUEUE,REPORTS,OUT,NAMES,verify_control
from variants.metric_tuning import run as engine


def with_velocity_default(reader):
    def read(path):
        result=reader(path)
        if Path(path).name=='metric_config.json':result.setdefault('velocity_source','independent')
        return result
    return read


def finalize():
    plan=verify_control(); lock=engine.read_json(QUEUE/'selection_lock.json')
    from variants.flow_comparison import export as exporter
    exporter.RUNS,exporter.OUT,exporter.REPORTS=QUEUE,OUT,REPORTS
    exporter.read=with_velocity_default(exporter.read)
    for directory in [CONTROL,TREATMENT]:
        for tag in ['best','50000']:(OUT/directory.parent.name/'pairs'/tag).mkdir(parents=True,exist_ok=True)
        exporter.export(directory)
    from variants.c1_flow_ablation import results as helpers
    motion=helpers.motion_module();motion.read=with_velocity_default(motion.read)
    manifest=motion.read(motion.CACHE/'manifest.json')
    assert all(motion.sha(motion.CACHE/k)==v for k,v in manifest['output_sha256'].items())
    rows=[motion.evaluate(d,tag,manifest) for d in [CONTROL,TREATMENT] for tag in ['best','50000']]
    engine.write_json(REPORTS/'motion_validation_reuse.json',{'status':'complete','results':rows,
        'historical_control_reused':True,'observation_manifest_sha256':motion.sha(motion.CACHE/'manifest.json')})
    validation={d.parent.name:engine.read_json(d/'metric_best.json')['PSNR']['metrics'] for d in [CONTROL,TREATMENT]}
    test={d.parent.name:engine.read_json(d/'results.json') for d in [CONTROL,TREATMENT]}
    metrics={r['model']:r['metrics'] for r in rows if r['snapshot']=='best'}
    a,b=NAMES
    deltas={'validation_PSNR_dB':validation[b]['PSNR']-validation[a]['PSNR'],
        'validation_LPIPS_relative':validation[b]['LPIPS']/validation[a]['LPIPS']-1,
        'geometry_EPE_relative':metrics[b]['geometric_EPE_px']/metrics[a]['geometric_EPE_px']-1,
        'velocity_EPE_relative':metrics[b]['velocity_EPE_px']/metrics[a]['velocity_EPE_px']-1}
    criteria={'image_PSNR':deltas['validation_PSNR_dB']>=-.2,
        'image_LPIPS':deltas['validation_LPIPS_relative']<=.05,
        'geometry_motion':deltas['geometry_EPE_relative']<=.05,
        'velocity_motion':deltas['velocity_EPE_relative']<=-.10}
    result={'historical_control_reused':True,'cancelled_O0_excluded':True,'validation':validation,'test':test,
        'motion':metrics,'deltas_O1_minus_historical_C1':deltas,'prospective_criteria':criteria,
        'composite_benefit':all(criteria.values()),'limitations':plan['limitations']}
    engine.write_json(REPORTS/'results_reuse.json',result)
    from reproduction.scripts.training_checkpoint import load_state
    proofs={}
    for directory in [CONTROL,TREATMENT]:
        checkpoint=directory/'checkpoints/resume.pt';receipt=engine.read_json(checkpoint.with_suffix('.json'))
        assert receipt['iteration']==50000 and receipt['sha256']==engine.sha256(checkpoint)
        full=load_state(checkpoint)
        def finite(value):
            if torch.is_tensor(value):return bool(value.isfinite().all())
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
            return True
        assert all(finite(full[k]) for k in ['gaussian_parameters','velocity_net','grid4d','deform','gaussian_optimizer','deform_optimizer'])
        g,d=motion.load(directory,'50000')
        for key,module in [('grid4d',d.grid4d),('deform',d.deform),('velocity_net',g.velocity_net)]:
            assert all(torch.equal(v.cpu(),module.state_dict()[k].cpu()) for k,v in full[key].items())
        assert all(torch.allclose(v.cpu(),getattr(g,k).detach().cpu(),atol=1e-7,rtol=1e-6) for k,v in full['gaussian_parameters'].items())
        snapshots={str(p.relative_to(directory)):engine.sha256(p) for tag in ['best','50000']
            for folder in ['point_cloud','deform'] for p in (directory/folder/('iteration_'+tag)).rglob('*') if p.is_file()}
        export=engine.read_json(REPORTS/(directory.parent.name+'_exports.json'))
        assert export['pairs']==100 and export['pixel_identity_verified'] and all(v['frames']==50 for v in export['videos'])
        proofs[directory.parent.name]={'full_checkpoint_sha256':receipt['sha256'],'full_snapshot_parity':True,
            'all_parameters_optimizer_finite':True,'snapshot_sha256':snapshots,
            'zip_sha256':engine.sha256(OUT/(directory.parent.name+'_GT_Render.zip'))}
        del full,g,d;torch.cuda.empty_cache()
    samples=[json.loads(s) for s in (TREATMENT/'camera_sampling.jsonl').read_text().splitlines()]
    assert [s['iteration'] for s in samples]==list(range(3001,50001))
    events=[json.loads(s) for s in (TREATMENT/'observed_flow.jsonl').read_text().splitlines()]
    due=[s for s in samples if s['iteration']%100==0]
    assert [s['iteration'] for s in events]==[s['iteration'] for s in due]
    assert all(e['source']==int(s['frame']) for e,s in zip(events,due))
    proof={'status':'passed','historical_control_reused':True,'historical_control_SHA_unchanged':True,
        'manual_review_pending':True,'same_actual_camera_schedule_claimed':False,
        'historical_camera_schedule_not_recorded':True,'models':proofs,'candidate_samples':len(samples),
        'flow_events':len(events),'flow_evaluations':sum('weighted_loss' in e for e in events),
        'flow_updates':sum('weighted_loss' in e and e['iteration']<50000 for e in events),
        'selection_lock_sha256':engine.sha256(QUEUE/'selection_lock.json')}
    engine.write_json(REPORTS/'final_verification_reuse.json',proof)
    helpers.NAMES=NAMES;helpers.review()
    lines=['# C1历史模型复用：光流开关比较','',
        '按用户纠正，原模型不重新训练；关闭组直接复用历史C1_repaired。新O0已取消并排除。仅新O1从相同完整3k状态到50k。',
        '原C1验证最佳47k与最终50k分别保留，不把测试最高50k冒称验证最佳。','',
        '|模型|验证最佳PSNR|测试best PSNR|测试final PSNR|几何EPE|速度推进EPE|',
        '|---|---:|---:|---:|---:|---:|']
    for n in NAMES:lines.append(f"|{n}|{validation[n]['PSNR']:.6f}|{test[n]['ours_best']['PSNR']:.6f}|{test[n]['ours_50000']['PSNR']:.6f}|{metrics[n]['geometric_EPE_px']:.6f}|{metrics[n]['velocity_EPE_px']:.6f}|")
    lines+=['','综合门槛通过：'+str(result['composite_benefit']),'变化：`'+json.dumps(deltas)+'`','',
        '历史复用、单场景单种子、CUDA非确定性与测试暴露限制因果结论。二维流为估计量，无3D物理真值。',
        '人工审阅待完成：早中晚、原帧85全尺寸PNG、解码图及视频。']
    (REPORTS/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    cells=''.join(f'<td><h2>{n}</h2><video id="v{i}" controls loop src="{n}/GT_Render_best.mp4"></video><img id="im{i}"></td>' for i,n in enumerate(NAMES))
    page='<!doctype html><meta charset="utf-8"><title>C1 光流比较</title><style>body{background:#15283c;color:white;font:16px system-ui;margin:24px}td{width:50%}img,video{width:100%}pre{white-space:pre-wrap}</style><h1>历史C1（无光流）与新O1（有光流）</h1><p>关闭组直接复用，未重训；每个画面左GT、右Render。</p><button onclick="v0.currentTime=v1.currentTime=0;v0.play();v1.play()">同步播放</button><button onclick="v0.pause();v1.pause()">暂停</button><table><tr>'+cells+'</tr></table><input id="f" type="range" min="0" max="49" value="21"><pre>'+json.dumps(result,ensure_ascii=False,indent=2)+'</pre><script>const names='+json.dumps(NAMES)+';function u(){names.forEach((n,i)=>document.getElementById("im"+i).src=n+"/pairs/best/"+String(f.value).padStart(5,"0")+".png")}f.oninput=u;u()</script>'
    (OUT/'index.html').write_text(page,encoding='utf8');verify_control()
    print('HISTORICAL C1 / NEW O1 COMPARISON VERIFIED; MANUAL REVIEW PENDING',flush=True)


if __name__=='__main__':finalize()
