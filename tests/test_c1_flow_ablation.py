"""Treatment isolation, exact C1 delegation, and data/schedule contracts."""
import json
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import torch
from variants.c1_flow_ablation.runtime import Tuning, Parent
from variants.c1_flow_ablation.run import configuration, NAMES, source_prefix


def tuning(tmp_path, enabled=False):
    c, _ = configuration(NAMES[int(enabled)], smoke=True)
    p = tmp_path/'config.json'; p.write_text(json.dumps(c))
    return Tuning(p, tmp_path)


def test_only_one_training_flag_differs():
    for smoke in (False, True):
        a, ta = configuration(NAMES[0], smoke); b, tb = configuration(NAMES[1], smoke)
        assert ta == tb
        assert {k for k in set(a)|set(b) if a.get(k) != b.get(k)} == {'observed_flow_enabled'}
        assert not a['observed_flow_enabled'] and b['observed_flow_enabled']
        assert 'use_flow = False' in ta
        assert a['kinematic_weight'] == 0 and a['velocity_source'] == 'independent'


def test_flow_disabled_does_not_touch_inputs_cache_or_RNG(tmp_path):
    t = tuning(tmp_path); t.config['record_sampling'] = False
    r = random.getstate(); n = np.random.get_state(); cpu = torch.get_rng_state().clone()
    assert t.observed_loss(100, None, None, None, None, None, None, None) is None
    assert t.flow_tensors == {} and not hasattr(t, 'flow_pairs')
    assert random.getstate() == r and np.array_equal(n[1], np.random.get_state()[1])
    assert torch.equal(torch.get_rng_state(), cpu)


def test_disabled_new_flag_inherits_exact_parent_methods():
    for name in ['image_loss', 'physics_time', 'time_step', 'coverage_loss', 'regularization',
            'after_scene', 'after_update', 'save', 'restore', 'evaluate', 'kinematic_loss']:
        assert getattr(Tuning, name) is getattr(Parent, name)


def test_flow_schedule_has_no_pre_prefix_supervision(tmp_path):
    t = tuning(tmp_path, True)
    assert not any(t.flow_due(i) for i in range(1, 21))
    assert t.flow_due(30) and not t.flow_due(31)
    t.config.update(flow_start=3000, flow_interval=100)
    assert not t.flow_due(3000) and t.flow_due(3100) and not t.flow_due(3101)


def test_global_flags_off_disables_new_loss(tmp_path):
    t = tuning(tmp_path, True)
    t.enabled = False; t.observed_flow_enabled = False
    assert t.observed_loss(100, None, None, None, None, None, None, None) is None


def test_initialization_keeps_C1_model_and_optimizer_recipe():
    c, text = configuration(NAMES[0])
    assert text == (source_prefix()/'model_config.py').read_text()
    assert c['mask_weight'] == .03 and c['foreground_rgb_weight'] == .1
    assert c['positive_color'] and c['shared_physics_time'] and c['timeline'] == 'frame_index'
    assert c['affine_weight'] == 0 and not c['coverage_density']


def test_raw_cache_uses_only_training_ids_and_positive_time():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'data/derived/unified_flow_train_only/manifest.json').read_text())
    train = set(manifest['train_frame_ids'])
    assert len(train) == 120 and manifest['heldout_RGB_used'] is False
    assert manifest['depth_teacher_used'] is False
    for pair in manifest['pairs']:
        assert pair['source'] in train and pair['target'] in train
        assert pair['dt'] > 0 and abs(pair['dt']-(pair['target']-pair['source'])/199) < 1e-12
