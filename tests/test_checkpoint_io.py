import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from pidg_ext import checkpoint


def test_extended_checkpoint_retries_locked_manifest_without_losing_state(tmp_path,monkeypatch):
    destination=tmp_path/'resume.pt'
    state={'format_version':1,'boundary':'after_optimizer_step_and_lr_update','iteration':25000,
           'optimizer':{'step':torch.tensor(123)},'rng':torch.get_rng_state()}
    physical={'force_optimizer':{'step':torch.tensor(456)},'material':torch.tensor([1.,10.])}
    def baseline_save(local,path):
        torch.save(state,path)
        manifest={'iteration':25000,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        Path(path).with_suffix('.json').write_text(json.dumps(manifest))
        return manifest
    monkeypatch.setattr(checkpoint.baseline,'save_training_state',baseline_save)
    monkeypatch.setattr(checkpoint.time,'sleep',lambda _:None)
    original=Path.replace;attempts=[]
    def locked(source,target):
        if Path(target).suffix=='.json':
            attempts.append(target)
            if len(attempts)<3:raise PermissionError('transient Windows reader lock')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',locked)
    ext=SimpleNamespace(enabled=True,checkpoint_state=lambda:physical)
    manifest=checkpoint.save_training_state({'extension':ext},destination)
    loaded=checkpoint.baseline.load_state(destination)
    assert len(attempts)==3 and manifest['extension_saved']
    assert json.loads(destination.with_suffix('.json').read_text())==manifest
    assert manifest['sha256']==hashlib.sha256(destination.read_bytes()).hexdigest()
    assert torch.equal(loaded['rng'],state['rng'])
    assert loaded['optimizer']['step']==123
    assert loaded['constitutive_extension']['force_optimizer']['step']==456


def test_persistent_lock_is_bounded_and_preserves_files(tmp_path,monkeypatch):
    source=tmp_path/'pending';source.write_text('new')
    destination=tmp_path/'published';destination.write_text('old')
    calls=[]
    def denied(*args):calls.append(1);raise PermissionError('persistent lock')
    monkeypatch.setattr(Path,'replace',denied)
    monkeypatch.setattr(checkpoint.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):checkpoint.replace_with_retry(source,destination,attempts=3)
    assert len(calls)==3 and source.read_text()=='new' and destination.read_text()=='old'


def test_flags_off_keeps_original_checkpoint_path(monkeypatch):
    expected={'original':True}
    monkeypatch.setattr(checkpoint.baseline,'save_training_state',lambda *args:expected)
    monkeypatch.setattr(checkpoint,'replace_with_retry',lambda *args:pytest.fail('new writer used with flags off'))
    assert checkpoint.save_training_state({'extension':SimpleNamespace(enabled=False)},'unused') is expected


def test_only_exact_registered_io_patch_can_resume(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import run_constitutive_experiment as runner
    from pidg_benchmark import write_json
    monkeypatch.setattr(runner,'PROJECT',tmp_path)
    expected={'config':{'loss_weight':.001},'extension_hashes':{'checkpoint.py':'new','constitutive.py':'same'}}
    old={'config':{'loss_weight':.001},'extension_hashes':{'checkpoint.py':'old','constitutive.py':'same'}}
    with pytest.raises(ValueError):runner.check_resume_protocol(old,expected)
    write_json(tmp_path/'experiments/checkpoint_io_hotfix.json',{'old_sha256':'old','new_sha256':'new'})
    assert runner.check_resume_protocol(old,expected)['registry_sha256']
    assert old['extension_hashes']['checkpoint.py']=='old'
    altered={**expected,'config':{'loss_weight':.002}}
    with pytest.raises(ValueError):runner.check_resume_protocol(old,altered)
    altered={**expected,'extension_hashes':{**expected['extension_hashes'],'constitutive.py':'changed'}}
    with pytest.raises(ValueError):runner.check_resume_protocol(old,altered)
    resumed={**old,'resume_extension_hashes':expected['extension_hashes']}
    assert runner.check_resume_protocol(resumed,expected) is None
