"""Verified image artifacts and unchanged independent V2 flow observations."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
from variants.flow_comparison import export as old
from variants.metric_tuning.run import read_json as read, write_json as write, sha256
from .run import ROOT, REPORTS, RUNS, NAME
OUT=ROOT/'output/momentum_repair'
CONTROL=ROOT/'runs/pidg_unified_flow/L1_shared_stress/rubber_duck'


def export(directory):
    OUT.mkdir(parents=True,exist_ok=True)
    old.REPORTS,old.RUNS,old.OUT=REPORTS,RUNS,OUT
    old.export(directory)


def report():
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    sys.path.insert(0,str(ROOT/'variants/momentum_guard/code'))
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    import inspect
    assert 'momentum_guard' in inspect.getfile(DeformModel)
    p=ROOT/'output/flow_comparison/evaluate_motion_v2.py'
    spec=importlib.util.spec_from_file_location('guard_motion_eval',p)
    motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
    lock=read(RUNS/'selection_lock.json');assert NAME in lock['models']
    manifest=read(motion.CACHE/'manifest.json')
    assert all(sha256(motion.CACHE/k)==v for k,v in manifest['output_sha256'].items())
    run=RUNS/NAME/'rubber_duck'
    results=[motion.evaluate(run,tag,manifest) for tag in ['best','50000']]
    write(REPORTS/'motion_validation.json',{'status':'complete','results':results,
        'manifest_sha256':sha256(motion.CACHE/'manifest.json'),'source':'same frozen V2 post-selection raw optical-flow observations, not physical GT'})
    reference=read(ROOT/'experiments/unified_flow/motion_validation.json')
    cm=next(r['metrics'] for r in reference['results'] if r['model']=='L1_shared_stress' and r['snapshot']=='best')
    mm=results[0]['metrics'];sel=lock['models'][NAME];cs=read(CONTROL/'metric_best.json')['PSNR']
    delta={'validation_PSNR':sel['value']-cs['value'],
        'validation_LPIPS_relative':sel['metrics']['LPIPS']/cs['metrics']['LPIPS']-1,
        'geometric_EPE_relative':mm['geometric_EPE_px']/cm['geometric_EPE_px']-1,
        'velocity_EPE_relative':mm['velocity_EPE_px']/cm['velocity_EPE_px']-1}
    image=delta['validation_PSNR']>=1 and delta['validation_LPIPS_relative']<=.05
    decision={'deltas_M1_minus_L1':delta,'image_recovery':image,
        'composite_benefit':image and delta['velocity_EPE_relative']<=-.1 and delta['geometric_EPE_relative']<=.05,
        'single_seed_scene':True,'historical_test_exposure':True,'no_further_automatic_full_training':True}
    write(REPORTS/'decision.json',decision)
    # Same deterministic training-time probes, including the true unscaled residual.
    from variants.unified_flow.runtime import Tuning as Parent
    probes=[]
    for directory in [CONTROL,run]:
        os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
        g,d=motion.load(directory,'best')
        tune=Parent(directory/'metric_config.json',REPORTS/'residual_probe'/directory.parent.name)
        tune.run.mkdir(parents=True,exist_ok=True)
        for fid in [2,83,155]:
            value=tune.momentum_loss(50000,g,d,torch.tensor([(fid-1)/199],device='cuda'))
            rec=json.loads((tune.run/'material_momentum.jsonl').read_text().splitlines()[-1])
            probes.append({'model':directory.parent.name,'training_frame':fid,'snapshot':'best',**rec})
        del g,d,tune,value;torch.cuda.empty_cache()
    write(REPORTS/'unweighted_momentum.json',{'probes':probes,'interpretation':'No clipping/rescaling of reported residual; no physical stress/acceleration GT'})
    prefix=torch.load(ROOT/'runs/pidg_unified_flow/shared_prefix/rubber_duck/checkpoints/resume.pt',map_location='cpu',weights_only=False)
    state=torch.load(run/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
    assert state['iteration']==50000
    assert all(v.isfinite().all() for key in ['deform','grid4d','velocity_net','gaussian_parameters'] for v in state[key].values())
    assert all(torch.equal(v,state['velocity_net'][k]) for k,v in prefix['velocity_net'].items())
    assert not torch.equal(state['deform']['shared_stress.weight'],prefix['deform']['shared_stress.weight'])
    guards=[json.loads(s) for s in (run/'momentum_gradient_guard.jsonl').read_text().splitlines()]
    for r in guards:
        assert all(v['guarded_momentum_norm']<=.1*v['image_norm']*(1+1e-5) for v in r['groups'].values())
    expected=list(range(8100,50001,100))
    assert [r['iteration'] for r in guards]==expected
    verification={'full_iteration':50000,'finite':True,'legacy_physics_unchanged':True,
        'shared_head_updated':True,'guard_events':len(guards),'all_shared_caps_satisfied':True,
        'full_checkpoint_sha256':sha256(run/'checkpoints/resume.pt')}
    write(REPORTS/'results.json',{'metrics':read(run/'results.json'),'control_metrics':read(CONTROL/'results.json'),
        'selection':sel,'control_selection':cs,'decision':decision,'motion':results,'verification':verification,
        'protocol':read(REPORTS/'plan.json')})
    lines=['# 动量梯度约束实验：M1 对照 L1','',
        '从相同完整3k状态到总50k。M1只改变动量梯度分配策略，未改变网络/差分步长/原始权重；共享形变梯度以同批RGB梯度10%为上限，私有应力梯度保持。',
        '这是显式一阶梯度修改；真实前向物理残差未缩小。验证选择后才导出新测试，历史测试已暴露。','',
        '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','',
        '图像质量、独立二维观测运动、内部结构一致性需分别解释。结构上零速度偏差不等于物理真值；梯度约束通过不等于动量残差或应力物理正确。',
        '详细图像指标见 results.json，独立运动见 motion_validation.json，真实未加权残差见 unweighted_momentum.json。',
        '本文件为自动报告，交付前仍需独立SHA/像素/视频/快照核验及人工图像审阅。','']
    (REPORTS/'assessment.md').write_text('\n'.join(lines),encoding='utf8')
