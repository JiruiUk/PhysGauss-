import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_constitutive_experiment as runner
from report_constitutive import assert_comparable
from pidg_benchmark import write_json


def test_official_budget_cannot_be_a_smoke_result():
    assert runner.budget(True,50000)['resolution_width']==1600
    assert runner.budget(False,300)['smoke_only']
    with pytest.raises(ValueError,match='50000'):runner.budget(True,300)
    protocol={'scene':'rubber_duck','iterations':50000,'resolution_width':1600,'seed':0,'profile':'official','smoke_only':False}
    assert_comparable(protocol,protocol.copy())
    with pytest.raises(ValueError,match='iterations'):assert_comparable({**protocol,'iterations':300},protocol)
    with pytest.raises(ValueError,match='scene'):assert_comparable({**protocol,'scene':'controlled_two_material'},protocol)


def test_existing_original_50k_baseline_is_reused_without_preparing_or_training(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'PROJECT',tmp_path)
    write_json(tmp_path/'experiments/configs/A0.json',{'experiment':'A0','scene':'rubber_duck'})
    metrics={'ours_best':{'PSNR':27.,'SSIM':.97,'LPIPS':.03,'MS-SSIM':.98}}
    run=tmp_path/'runs/pidg_official/rubber_duck'
    write_json(run/'results.json',metrics)
    write_json(run/'reproduction.json',{'status':'completed','metrics':metrics,**runner.budget(True,50000)})
    monkeypatch.setattr(runner,'prepare',lambda:pytest.fail('Original baseline must not be retrained'))
    assert runner.run('A0','unused',50000,True)==run
    assert not (tmp_path/'runs/pidg_constitutive').exists()


def test_A5_diagnostics_do_not_change_A4_training_configuration():
    from pidg_benchmark import read_json
    root=Path(__file__).resolve().parents[1]/'experiments/configs'
    for left,right in [('A4_controlled','A5_controlled'),('A4_single','A5_single')]:
        a=read_json(root/(left+'.json'));b=read_json(root/(right+'.json'))
        for value in (a,b):
            value.pop('experiment');value.pop('identifiability',None)
        assert a==b


def test_matrix_selected_scope_remains_one_official_scene():
    import run_constitutive_matrix as worker
    plan={'execution':{'selected_long_configs':['A1','A2a','A2b','A4_single','A6_single'],'selected_long_scenes':['rubber_duck']}}
    jobs=worker.jobs(plan,'long')
    assert len(jobs)==5 and {j['scene'] for j in jobs}=={'rubber_duck'}
    plan['execution']['selected_long_scenes']=['mechanics_cloth']
    with pytest.raises(ValueError,match='outside'):worker.jobs(plan,'long')


def test_new_controller_forwards_stop_once_and_does_not_recreate_consumed_request(tmp_path,monkeypatch):
    queue=tmp_path/'queue';queue.mkdir();run=tmp_path/'run';run.mkdir()
    monkeypatch.setattr(runner,'RUNS',queue);monkeypatch.setattr(runner.time,'sleep',lambda _:None)
    (queue/'stop_queue.request').write_text('stop')
    request=run/'checkpoints/save_and_stop.request'
    class Child:
        pid=123;returncode=75;count=0
        def poll(self):
            self.count+=1
            if self.count==2:request.unlink()
            return None if self.count<3 else 75
    monkeypatch.setattr(runner.subprocess,'Popen',lambda *args,**kwargs:Child())
    assert runner.execute(['fake'],run,'train',tmp_path,{}, {'commands':[],'experiment':'A1','configuration_id':'A1'})==75
    assert not request.exists()


def test_saved_training_resume_uses_verified_checkpoint_and_preserves_outputs(tmp_path,monkeypatch):
    import hashlib
    from pidg_benchmark import read_json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'reproduction/scripts'))
    monkeypatch.setattr(runner,'PROJECT',tmp_path);monkeypatch.setattr(runner,'RUNS',tmp_path/'runs/pidg_constitutive')
    repo=tmp_path/'variant';original=tmp_path/'third_party/PIDG'
    for folder in (repo,original):
        for name in ('train_pidg.py','render.py','metrics.py','hashencoder/src/hashencoder.cu','hashencoder/src/hashencoder.h','hashencoder/src/bindings.cpp'):
            file=folder/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('fixture')
    helper=tmp_path/'reproduction/scripts/training_checkpoint.py';helper.parent.mkdir(parents=True);helper.write_text('fixture')
    data=tmp_path/'data/PIDG/rubber_duck';data.mkdir(parents=True)
    for name in ('transforms_train.json','transforms_test.json','points3d.ply','audit.json','download_manifest.json'):(data/name).write_text('{}')
    write_json(tmp_path/'experiments/configs/A1.json',{'experiment':'A1','scene':'rubber_duck','deformation_gradient':True})
    monkeypatch.setattr(runner,'prepare',lambda:repo)
    commands=[]
    def stopped(command,directory,step,*args):
        commands.append(command)
        assert step=='train'
        cp=directory/'checkpoints/resume.pt';cp.parent.mkdir(exist_ok=True);cp.write_bytes(b'controlled test checkpoint')
        write_json(cp.with_suffix('.json'),{'iteration':100,'sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'optimizer_saved':True,'rng_saved':True,'extension_saved':True})
        return 75
    monkeypatch.setattr(runner,'execute',stopped)
    with pytest.raises(SystemExit) as first:runner.run('A1','test')
    assert first.value.code==75
    directory=tmp_path/'runs/pidg_constitutive/test/rubber_duck'
    assert read_json(directory/'reproduction.json')['status']=='saved_stopped'
    with pytest.raises(SystemExit) as second:runner.run('A1','test',resume=True)
    assert second.value.code==75 and '--resume_checkpoint' in commands[-1]
    assert '--resume_checkpoint' not in commands[0]
