import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
from variants.unified_flow.model import stress_head, configure_from_env, optimizer_groups, predict_stress, latent
from variants.unified_flow.checkpoint import extend_state

ROOT = Path(__file__).resolve().parents[1]


def test_head_initialization_preserves_rng_and_is_reproducible():
    before = torch.random.get_rng_state().clone()
    a, b = stress_head(16), stress_head(16)
    assert torch.equal(before, torch.random.get_rng_state())
    assert a.weight.shape == (6, 16) and torch.equal(a.weight, b.weight)
    assert a.bias.count_nonzero() == 0 and a.weight.std() > 0


def test_checkpoint_extension_preserves_old_state_and_adds_empty_group():
    matrix = torch.randn(7, 16)
    old_moment = torch.randn(2)
    state = {'deform': {'gaussian_warp.weight': matrix},
        'deform_optimizer': {'param_groups': [{'params': [0, 1], 'lr': .002, 'name': 'deform'}, {'params': [2], 'lr': .02, 'name': 'grid'}],
                            'state': {0: {'exp_avg': old_moment}}},
        'opt': {'velocity_net_lr': .001}, 'dataset': {'network_args': {'width': 16}}}
    before = torch.random.get_rng_state().clone()
    additions = extend_state(state)
    assert torch.equal(before, torch.random.get_rng_state())
    assert state['deform']['gaussian_warp.weight'] is matrix
    assert state['deform_optimizer']['state'][0]['exp_avg'] is old_moment
    assert state['deform_optimizer']['param_groups'][-1] == {'params': [3, 4], 'lr': .001, 'name': 'shared_stress'}
    assert set(additions) == {'shared_stress.weight', 'shared_stress.bias'}
    assert set(state['deform_optimizer']['state']) == {0}


def test_shared_mode_freezes_old_network_and_off_preserves_it(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('unified_test_velocity', ROOT/'variants/flow_comparison/code/motion_utils/pidg.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    path = tmp_path/'config.json'; monkeypatch.setenv('PIDG_METRIC_CONFIG', str(path))
    net = module.VelocityNet(max_particles=32)
    path.write_text(json.dumps({'enabled': False, 'stress_source': 'shared'}))
    configure_from_env(net); assert all(p.requires_grad for p in net.parameters())
    path.write_text(json.dumps({'enabled': True, 'stress_source': 'shared', 'velocity_source': 'trajectory'}))
    configure_from_env(net); assert not any(p.requires_grad for p in net.parameters())


@pytest.mark.skipif(os.environ.get('PIDG_UNIFIED_GPU_TEST') != '1', reason='explicit shared-grid CUDA check')
def test_shared_trunk_gradients_snapshot_optimizer_and_flags_off(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(ROOT/'variants/unified_flow/code'))
    monkeypatch.setenv('PIDG_HASH_SOURCE_DIR', str(ROOT/'third_party/PIDG/hashencoder'))
    monkeypatch.setenv('PIDG_HASH_BUILD_DIR', str(ROOT/'third_party/PIDG/tmp_build'))
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'enabled': True, 'stress_source': 'shared'}))
    monkeypatch.setenv('PIDG_METRIC_CONFIG', str(path))
    from scene_PIDG.deform_model import DeformModel
    from scene_PIDG.network import DeformNetwork
    spec = importlib.util.spec_from_file_location('old_deform_network', ROOT/'variants/flow_comparison/code/scene_PIDG/network.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    parent = module.DeformNetwork(4, 12, width=16, depth=1).cuda()
    off = DeformNetwork(4, 12, width=16, depth=1).cuda()
    off.load_state_dict(parent.state_dict(), strict=True)
    spatial, temporal = torch.randn(8, 4, device='cuda'), torch.randn(8, 12, device='cuda')
    assert all(torch.equal(a, b) for a, b in zip(parent(spatial, temporal), off(spatial, temporal)))
    grid = dict(canonical_num_levels=2, canonical_level_dim=2, canonical_base_resolution=2,
        canonical_desired_resolution=4, canonical_log2_hashmap_size=5, deform_num_levels=2,
        deform_level_dim=2, deform_base_resolution=[2, 2, 2], deform_desired_resolution=[4, 4, 4], deform_log2_hashmap_size=5)
    net = dict(width=16, depth=1, unified_stress_head=True)
    deform = DeformModel(grid, net, reg_spatial_able=False, reg_temporal_able=False)
    opt = SimpleNamespace(network_lr_scale=5., grid_lr_scale=50., position_lr_init=.00016,
        position_lr_final=.0000016, position_lr_delay_mult=.01, deform_lr_max_steps=45000, velocity_net_lr=.001)
    deform.train_setting(opt)
    assert [g['name'] for g in deform.optimizer.param_groups] == ['deform', 'grid', 'shared_stress']
    head_ids = {id(p) for p in deform.deform.shared_stress.parameters()}
    assert not head_ids & {id(p) for p in deform.optimizer.param_groups[0]['params']}
    x = torch.rand(24, 3, device='cuda')*.2; t = .4
    observed = []
    hook = deform.deform.gaussian_warp.register_forward_pre_hook(lambda m, args: observed.append(args[0]))
    deform.step(x, x.new_full((len(x), 1), t)); hook.remove()
    coords = torch.cat([x*deform.scale_xyz[None], x.new_full((len(x), 1), (t*2*deform.grid4d.bound-deform.grid4d.bound)*.9)], -1)
    sp, tp = deform.grid4d(coords)
    assert torch.equal(observed[0], latent(deform.deform, sp, tp))
    sigma = predict_stress(deform, x, t)
    assert torch.equal(sigma, deform.deform.shared_stress(observed[0]))
    before = {k: v.clone() for k, v in deform.deform.state_dict().items()}
    (sigma-torch.arange(6, device='cuda')).square().mean().backward()
    for component in [deform.grid4d, deform.deform.spatial_mlp, deform.deform.temporal_mlp, deform.deform.shared_stress]:
        gradients = [p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(g.isfinite().all() for g in gradients) and sum(g.abs().sum() for g in gradients)>0
    deform.optimizer.step()
    assert not torch.equal(before['shared_stress.weight'], deform.deform.shared_stress.weight)
    assert torch.equal(before['gaussian_warp.weight'], deform.deform.gaussian_warp.weight)
    deform.save_weights(str(tmp_path/'snapshot'), 1)
    other = DeformModel(grid, net, reg_spatial_able=False, reg_temporal_able=False)
    other.load_weights(str(tmp_path/'snapshot'), 1); other.train_setting(opt)
    other.optimizer.load_state_dict(deform.optimizer.state_dict())
    assert all(torch.equal(v, other.deform.state_dict()[k]) for k, v in deform.deform.state_dict().items())
    assert set(other.optimizer.state_dict()['state']) == set(deform.optimizer.state_dict()['state'])
