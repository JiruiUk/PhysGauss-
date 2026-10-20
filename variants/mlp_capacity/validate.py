"""GPU architecture/gradient validation plus existing flags-off regression test."""
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import variants, resolved_config, BASE_CONFIG, REPO, REPORTS
sys.path.insert(0, str(ROOT / 'scripts'))
from pidg_benchmark import runtime_hashes, sha256, write_json

if __name__ == '__main__':
    import torch
    frozen = runtime_hashes(REPO)
    os.environ['PIDG_HASH_SOURCE_DIR'] = str(REPO / 'hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR'] = str(REPO / 'tmp_build')
    sys.path.insert(0, str(REPO))
    from scene_PIDG.network import DeformNetwork
    base = runpy.run_path(str(BASE_CONFIG))
    assert base['network_args'] == {'depth': 0, 'width': 256, 'directional': True}
    records = []
    for item in variants():
        config = {}
        exec(compile(resolved_config(item), '<capacity-config>', 'exec'), config)
        for key, value in base.items():
            if not key.startswith('__') and key != 'network_args':
                assert config[key] == value, key
        assert config['network_args'] == dict(base['network_args'], width=item['width'], depth=item['depth'])
        torch.manual_seed(0)
        model = DeformNetwork(32, 192, **config['network_args']).cuda()
        count = sum(p.numel() for p in model.parameters())
        assert count == item['mlp_parameters'], (count, item)
        space = torch.randn(8, 32, device='cuda', requires_grad=True)
        temporal = torch.randn(8, 192, device='cuda', requires_grad=True)
        outputs = model(space, temporal)
        assert [tuple(x.shape) for x in outputs] == [(8, 7), (8, 4), (8, 3)]
        sum(x.square().mean() for x in outputs).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert space.grad.norm() > 0 and temporal.grad.norm() > 0
        records.append({**item, 'actual_mlp_parameters': count, 'finite_gradients': True,
                        'only_width_depth_changed': True})
        del model, outputs, space, temporal
        torch.cuda.empty_cache()
    subprocess.run([sys.executable, '-m', 'pytest', '-q',
                    'tests/test_pidg_constitutive.py::test_disabled_extension_is_a_noop_and_does_not_consume_rng',
                    'tests/test_pidg_constitutive.py::test_graph_cache_does_not_consume_training_rng_and_refreshes_on_topology_change'],
                   cwd=ROOT, check=True)
    assert runtime_hashes(REPO) == frozen
    receipt = {'status': 'passed', 'architecture_checks': records, 'existing_regression_tests': 2,
               'baseline_source_unchanged': True, 'baseline_runtime_hashes': frozen,
               'validator_sha256': sha256(__file__), 'torch': str(torch.__version__)}
    write_json(REPORTS / 'validation.json', receipt)
    print(receipt['status'], records, flush=True)
