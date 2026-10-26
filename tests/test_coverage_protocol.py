import torch
import pytest
from variants.coverage_tuning.run import configuration,NAMES
from variants.v3_iso_scratch.run import configuration as original, NAMES as ORIGINAL


def test_additions_are_ordered_and_initial_optimizer_unchanged():
    baseline,text = original(ORIGINAL[0])
    for i,n in enumerate(NAMES):
        c,t = configuration(n)
        assert t == text
        assert c['isotropic_weight'] == c['kinematic_weight'] == 0
        assert c['parent_run'] == ''
        assert c['mask_weight'] == (.1 if i>=1 else 0)
        assert c['affine_weight'] == (1e-4 if i>=2 else 0)
        assert c['coverage_density'] == (i>=3)
        assert c['coverage_start'] > 15000


def test_fork_only_adds_guarded_hooks_to_training_source():
    from variants.coverage_tuning.prepare import prepare
    from variants.v3_iso_scratch.prepare import prepare as parent
    text = (prepare()/'train_pidg.py').read_text(encoding='utf-8')
    text = text.replace('from variants.coverage_tuning.runtime import Tuning',
                        'from variants.v3_iso_scratch.runtime import Tuning')
    text = text.replace('        coverage_loss = tuning.coverage_loss(iteration, viewpoint_cam, render_pkg_re, gaussians, d_xyz)\n'
        '        if coverage_loss is not None:\n            loss = loss + coverage_loss\n\n','')
    text = text.replace('            tuning.after_update(iteration, gaussians)\n','')
    assert text == (parent()/'train_pidg.py').read_text(encoding='utf-8')


def test_neighbor_constraint_cannot_silently_apply_to_multiple_objects(tmp_path):
    import json
    from variants.coverage_tuning.runtime import Tuning
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'enabled':True,'affine_weight':1e-4,'single_object':False}))
    with pytest.raises(ValueError,match='object IDs'): Tuning(path,tmp_path)
