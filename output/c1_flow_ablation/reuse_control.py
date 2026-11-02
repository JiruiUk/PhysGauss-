"""User-corrected queue: reuse historical C1; train ONLY the optical-flow arm."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from variants.c1_flow_ablation import run as original
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base

QUEUE = original.RUNS/'reuse_control_queue'
CONTROL = ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
TREATMENT = original.RUNS/'O1_C1_flow_on/rubber_duck'
REPORTS, OUT = original.REPORTS, original.OUT
NAMES = ['C1_repaired', 'O1_C1_flow_on']


def control_hashes():
    files = [CONTROL/p for p in ['metric_config.json','model_config.py','metric_best.json',
        'results.json','per_view.json','branch.json','checkpoints/resume.pt','checkpoints/resume.json']]
    files += [CONTROL.parents[1]/'selection_lock.json']
    for tag in ['best','50000']:
        for folder in ['point_cloud','deform']:
            files += [p for p in (CONTROL/folder/('iteration_'+tag)).rglob('*') if p.is_file()]
        for kind in ['gt','renders']:
            files += list((CONTROL/'test'/('ours_'+tag)/kind).glob('*.png'))
    return {str(p.relative_to(ROOT)):engine.sha256(p) for p in files}


def verify_control():
    plan = engine.read_json(REPORTS/'reuse_control_plan.json')
    assert control_hashes() == plan['historical_control_sha256']
    assert engine.read_json(CONTROL/'branch.json')['identical_full_checkpoint_sha256'] == plan['common_3k_sha256']
    assert engine.read_json(TREATMENT/'branch.json')['identical_full_checkpoint_sha256'] == plan['common_3k_sha256']
    assert (CONTROL/'model_config.py').read_text() == (TREATMENT/'model_config.py').read_text()
    assert engine.read_json(CONTROL/'metric_best.json')['PSNR'] == plan['historical_validation_selection']
    return plan


def write_plan():
    path = REPORTS/'reuse_control_plan.json'
    if path.exists(): return verify_control()
    cancelled = original.RUNS/'O0_C1_flow_off/rubber_duck'
    receipt = engine.read_json(cancelled/'checkpoints/resume.json')
    assert engine.sha256(cancelled/'checkpoints/resume.pt') == receipt['sha256']
    plan = {'user_correction':'Do not retrain the original model; reuse historical C1 as optical-flow-off control.',
        'created_utc':engine.now(), 'historical_control':str(CONTROL), 'only_full_training_model':str(TREATMENT),
        'historical_validation_selection':engine.read_json(CONTROL/'metric_best.json')['PSNR'],
        'historical_control_sha256':control_hashes(),
        'common_3k_sha256':engine.read_json(TREATMENT/'branch.json')['identical_full_checkpoint_sha256'],
        'cancelled_O0':{'saved_iteration':receipt['iteration'],'checkpoint_sha256':receipt['sha256'],
            'preserved':True,'never_resume':True,'not_used_as_control':True},
        'new_treatment_steps':47000,'total_treatment_steps':50000,'no_new_control_training':True,
        'original_frozen_protocol_sha256':engine.sha256(REPORTS/'plan.json'),
        'original_source_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json'),
        'comparison':'O1 versus historical C1 validation-best; separately final50k versus final50k',
        'limitations':['historical control reuse, not a fresh simultaneous two-run comparison',
            'single scene and seed','GPU nondeterminism','historical test exposure'],
        'benefit_criteria_unchanged':True,'no_more_full_candidates':True}
    engine.write_json(path,plan)
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=16)
    parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    QUEUE.mkdir(parents=True,exist_ok=True)
    repo=original.prepare(); original.wire(repo)
    # Keep the old queue's explicit stop request intact. Its original main remains blocked.
    engine.RUNS=QUEUE; base.RUNS=QUEUE
    original.assert_frozen(repo); plan=write_plan()
    assert engine.read_json(REPORTS/'preflight.json')['status']=='passed'
    if args.prepare_only: print('HISTORICAL C1 VERIFIED; ONLY O1 MAY TRAIN',flush=True);return
    deadline=datetime.now(timezone.utc)+timedelta(hours=args.hours)
    state={'pid':os.getpid(),'status':'training','started_utc':engine.now(),'deadline_utc':deadline.isoformat(),
        'control_reused':str(CONTROL),'only_training':str(TREATMENT)}
    with engine.lock():
        engine.write_json(QUEUE/'controller.json',state)
        os.environ['PIDG_EXPORT_ALPHA']='1'
        try:
            if original.parent.step(TREATMENT)<50000:
                if base.train(TREATMENT,repo,deadline)==75: raise SystemExit(75)
            assert original.parent.step(TREATMENT)==50000
            verify_control()
            content={'selection_split':'validation','models':{
                NAMES[0]:plan['historical_validation_selection'],
                NAMES[1]:engine.read_json(TREATMENT/'metric_best.json')['PSNR']},
                'historical_control_reused':True}
            lock=QUEUE/'selection_lock.json'
            if lock.exists():
                saved=engine.read_json(lock);assert all(saved[k]==v for k,v in content.items())
            else: engine.write_json(lock,{**content,'locked_utc':engine.now()})
            state['status']='rendering_treatment';engine.write_json(QUEUE/'controller.json',state)
            base.render(TREATMENT,repo,deadline,tags=['best','50000'])
            state['status']='postprocess';engine.write_json(QUEUE/'controller.json',state)
            from output.c1_flow_ablation.compare_reuse import finalize
            finalize(); original.assert_frozen(repo);verify_control()
            state['status']='completed'
        except SystemExit as error:
            state['status']='saved_stopped' if error.code==75 else 'failed'
            if error.code!=75:raise
        except Exception as error:
            state.update(status='failed',error=str(error));raise
        finally:
            state['updated_utc']=engine.now();engine.write_json(QUEUE/'controller.json',state)


if __name__=='__main__':main()
