import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_upgrade
import snapshot_upgrade
from pidg_benchmark import write_json,read_json
from upgrade_configs import stage_configs
from upgrade_scope import selected_scenes
from argparse import Namespace


def test_each_stage_adds_expected_switches():
    configs=stage_configs()
    assert set(configs)==set(range(1,16))
    assert not configs[2].get('use_constitutive_model',False)
    assert configs[4]['anchor_deformation'] and configs[4]['spatial_bandlimit']==2
    assert configs[8]['momentum_mode']=='closed_strong'
    assert configs[9]['momentum_mode']=='weak'
    assert configs[13]['use_uncertainty_flow']
    assert configs[15]['motion_source']=='bspline' and configs[15]['use_rollout']


def test_concurrent_report_writers_are_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path=tmp_path/'report.json'
    write_json(path,{'sequence':-1,'payload':'x'*10000})
    def write(sequence):
        write_json(path,{'sequence':sequence,'payload':str(sequence)*10000})
        data=read_json(path)
        assert 'sequence' in data and 'payload' in data
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write,range(24)))
    assert 0<=read_json(path)['sequence']<24
    assert not list(tmp_path.glob('*.tmp'))


def test_reader_tolerates_a_manifest_producer_finishing_its_write(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    path=tmp_path/'resume.json'
    path.write_text('{')
    def finish_manifest():
        time.sleep(.02)
        write_json(path,{'iteration':1000,'optimizer_saved':True})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(finish_manifest)
        assert read_json(path)['iteration']==1000
        future.result()


def test_exclusive_controller_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(run_upgrade,'QUEUE',tmp_path)
    with run_upgrade.QueueLock():
        with pytest.raises(OSError):
            with run_upgrade.QueueLock(): pass
    with run_upgrade.QueueLock(): pass


def test_stop_request_is_not_recreated_after_trainer_consumes_it(tmp_path,monkeypatch):
    queue=tmp_path/'queue';queue.mkdir()
    run=tmp_path/'run';(run/'checkpoints').mkdir(parents=True)
    (queue/'stop_queue.request').write_text('save and stop')
    request=run/'checkpoints/save_and_stop.request'
    monkeypatch.setattr(run_upgrade,'QUEUE',queue)
    monkeypatch.setattr(run_upgrade.time,'sleep',lambda _:None)
    writes=[]
    original=Path.write_text
    def record_write(path,*args,**kwargs):
        if path==request: writes.append(1)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',record_write)
    class Child:
        pid=123
        returncode=75
        count=0
        def poll(self):
            self.count+=1
            if self.count==2: request.unlink()
            return None if self.count<3 else 75
    monkeypatch.setattr(run_upgrade.subprocess,'Popen',lambda *args,**kwargs:Child())
    assert run_upgrade.run_command(['fake'],tmp_path,run,'train',{})==75
    assert len(writes)==1 and not request.exists()


def make_completed(path,iterations=50000,psnr=20.):
    metrics={'ours_best':{'PSNR':psnr,'SSIM':.8,'LPIPS':.2,'MS-SSIM':.75}}
    write_json(path/'results.json',metrics)
    write_json(path/'reproduction.json',{'status':'completed','metrics':metrics,
               'iterations':iterations,'resolution_width':1600,'seed':0,'profile':'official'})


def test_long_report_never_counts_queued_runs_or_cross_budget_gains(tmp_path,monkeypatch):
    monkeypatch.setattr(run_upgrade,'PROJECT',tmp_path)
    queue=tmp_path/'queue';queue.mkdir();monkeypatch.setattr(run_upgrade,'QUEUE',queue)
    baseline=tmp_path/'runs/pidg_official/mechanics_cloth'
    candidate=tmp_path/'variants/phase01_continuous_time/runs/long/mechanics_cloth'
    make_completed(baseline,psnr=20.)
    make_completed(candidate,psnr=21.)
    run_upgrade.report()
    report=read_json(queue/'results.json')
    assert len(report['rows'])==75 and not report['all_completed']
    assert report['rows'][0]['delta']['PSNR']==1
    assert all('delta' not in row for row in report['rows'][1:])
    make_completed(candidate,iterations=120)
    with pytest.raises(ValueError,match='protocol mismatch'):
        run_upgrade.report()


def test_snapshot_refuses_to_overwrite_sources(tmp_path,monkeypatch):
    monkeypatch.setattr(snapshot_upgrade,'PROJECT',tmp_path)
    source=tmp_path/'variants/development/code';source.mkdir(parents=True)
    for name in ('train_pidg.py','render.py','metrics.py'):
        (source/name).write_text('# source\n')
    variant=snapshot_upgrade.snapshot(2,stage_configs()[2])
    assert read_json(variant/'stage.json')['long_training']=='pending'
    with pytest.raises(FileExistsError): snapshot_upgrade.snapshot(2,stage_configs()[2])
    assert (variant/'code/train_pidg.py').read_text()=='# source\n'


def test_selected_scene_report_ignores_excluded_runs(tmp_path,monkeypatch):
    monkeypatch.setattr(run_upgrade,'PROJECT',tmp_path)
    queue=tmp_path/'queue';queue.mkdir();monkeypatch.setattr(run_upgrade,'QUEUE',queue)
    write_json(queue/'benchmark_scope.json',{'scenes':['mechanics_cloth']})
    make_completed(tmp_path/'runs/pidg_official/mechanics_cloth',psnr=20.)
    for phase,name in run_upgrade.NAMES.items():
        variant=tmp_path/f'variants/phase{phase:02d}_{name}'
        make_completed(variant/'runs/long/mechanics_cloth',psnr=21.)
        write_json(variant/'runs/long/balls_reaction/reproduction.json',{'status':'failed'})
    run_upgrade.report()
    report=read_json(queue/'results.json')
    assert len(report['rows'])==15 and report['all_completed']
    assert report['scenes']==['mechanics_cloth'] and len(report['excluded_scenes'])==4
    assert all(row['scene']=='mechanics_cloth' and row['delta']['PSNR']==1 for row in report['rows'])


def test_queue_uses_persisted_scope_for_execution_and_phase_completion(tmp_path,monkeypatch):
    monkeypatch.setattr(run_upgrade,'PROJECT',tmp_path)
    queue=tmp_path/'queue';queue.mkdir();monkeypatch.setattr(run_upgrade,'QUEUE',queue)
    write_json(queue/'benchmark_scope.json',{'scenes':['mechanics_cloth']})
    variant=tmp_path/'variants/phase01_continuous_time'
    write_json(variant/'stage.json',{'phase':1,'unit_tests':'passed','smoke':'completed'})
    calls=[]
    def fake_run(variant,scene,mode):
        calls.append((scene,mode))
        make_completed(variant/'runs'/mode/scene)
    monkeypatch.setattr(run_upgrade,'run_scene',fake_run)
    monkeypatch.setattr(run_upgrade,'report',lambda:None)
    run_upgrade.execute_queue(Namespace(stages=[1],mode='long',scenes=None))
    assert calls==[('mechanics_cloth','long')]
    stage=read_json(variant/'stage.json')
    assert stage['long_training']=='completed' and stage['long_training_scenes']==['mechanics_cloth']
    assert read_json(queue/'controller.json')['status']=='completed'
    assert read_json(queue/'controller.json')['scenes']==['mechanics_cloth']


@pytest.mark.parametrize('scenes',[[],['unknown'],['mechanics_cloth','mechanics_cloth'],'mechanics_cloth'])
def test_invalid_scope_never_falls_back_to_five_scenes(tmp_path,scenes):
    write_json(tmp_path/'benchmark_scope.json',{'scenes':scenes})
    with pytest.raises(ValueError,match='Invalid benchmark scope'):
        selected_scenes(tmp_path)
