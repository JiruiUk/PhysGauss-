import pytest
from variants.integrity_continuation import run


def fixture(tmp_path,iteration=20000):
    (tmp_path/'checkpoints').mkdir();p=tmp_path/'checkpoints/resume.pt';p.write_bytes(b'full-state-contract-fixture')
    digest=run.engine.sha256(p);config={'coverage_density':False,'fixed_original_recipe':True}
    run.engine.write_json(p.with_suffix('.json'),{'sha256':digest,'iteration':iteration,'optimizer_saved':True,'rng_saved':True})
    run.engine.write_json(p.with_suffix('.metrics.json'),{'checkpoint_sha256':digest,'iteration':iteration,'config':config})
    run.engine.write_json(tmp_path/'metric_config.json',config);(tmp_path/'model_config.py').write_text('iterations=50000\n')
    return p


def test_only_four_incomplete_pilots_are_targeted():
    assert [j['id'] for j in run.JOBS]==['H1_balanced_coverage','H4_color_density','F1_feedback','F2_conservative_retry']
    assert not set(j['id'] for j in run.JOBS)&set(run.REFERENCES)


def test_full_state_is_verified_and_finished_checkpoint_skips_training(tmp_path):
    fixture(tmp_path,50000)
    assert not run.verify_checkpoint(tmp_path)['needs_training']


def test_saved_configuration_change_is_rejected(tmp_path):
    fixture(tmp_path)
    run.engine.write_json(tmp_path/'metric_config.json',{'coverage_density':False,'fixed_original_recipe':False})
    with pytest.raises(ValueError,match='config/selection'):run.verify_checkpoint(tmp_path)


def test_corrupted_tensor_state_is_rejected(tmp_path):
    p=fixture(tmp_path);p.write_bytes(b'changed weights')
    with pytest.raises(ValueError,match='Invalid continuation'):run.verify_checkpoint(tmp_path)
