import json
import pytest
from variants.scratch_integrity.stage import schedule,legacy_iteration,verify_binding_boundary


def test_activation_is_after_common_topology_boundary():
    assert schedule(15000,15000,10,2000)==(0.,False)
    assert schedule(15001,15000,10,2000)==(.0005,False)
    assert schedule(15010,15000,10,2000)==(.005,True)
    assert schedule(17000,15000,10,2000)==(1.,True)


def test_no_early_activation_or_zero_ramp():
    assert schedule(100,15000,10,2000)==(0.,False)
    for args in [(15001,14999,10,2000),(15001,15000,0,2000),(15001,15000,10,0)]:
        with pytest.raises(ValueError):schedule(*args)


def test_sampling_restarts_at_new_activation_instead_of_old_50k():
    assert legacy_iteration(15010,15000)==50010
    assert legacy_iteration(20020,20000)==50020
    with pytest.raises(ValueError):legacy_iteration(14990,15000)


def test_binding_rejects_wrong_prefix_or_topology():
    verify_binding_boundary(15000,15000,90000,90000)
    for args in [(15000,50000,90000,90000),(15000,15000,90001,90000),(14999,14999,90000,90000)]:
        with pytest.raises(ValueError):verify_binding_boundary(*args)


def test_disabled_runtime_delegates_without_binding(tmp_path,monkeypatch):
    from variants.scratch_integrity.runtime import Tuning,Color
    p=tmp_path/'config.json';p.write_text(json.dumps({'enabled':True,'transfer_integrity_enabled':False}))
    t=Tuning(p,tmp_path);sentinel=object()
    monkeypatch.setattr(Color,'coverage_loss',lambda *args:sentinel)
    assert t.coverage_loss(15010,None,None,None,None) is sentinel
    assert not hasattr(t,'binding')


def test_image_physics_optimizer_paths_are_inherited():
    from variants.scratch_integrity.runtime import Tuning,Color
    for name in ['image_loss','physics_time','time_step','regularization','kinematic_loss','after_update','evaluate']:
        assert getattr(Tuning,name) is getattr(Color,name)


def test_before_start_does_not_evaluate_new_geometry(tmp_path,monkeypatch):
    from variants.scratch_integrity.runtime import Tuning,Color
    p=tmp_path/'config.json';p.write_text(json.dumps({'enabled':True,'transfer_integrity_enabled':True,'transfer_start':15000}))
    t=Tuning(p,tmp_path);sentinel=object()
    monkeypatch.setattr(Color,'coverage_loss',lambda *args:sentinel)
    monkeypatch.setattr(t,'geometry_loss',lambda *args:pytest.fail('Early geometry access'))
    assert t.coverage_loss(15000,None,None,None,None) is sentinel


def test_prefix_is_cold_and_topology_stable():
    from variants.scratch_integrity.prefix import plan,RECIPE,engine
    import runpy
    c=engine.read_json(RECIPE/'metric_config.json')
    assert c['parent_run']=='' and not c['coverage_density']
    assert runpy.run_path(str(RECIPE/'model_config.py'))['densify_until_iter']<=min(plan()['common_prefix_candidates'])
    assert plan()['total_actual_optimizer_updates_each']==50000
