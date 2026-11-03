"""Default-off native value parity and real optimizer gradient smoke, not formal data."""
from pathlib import Path
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.synthetic_audit.input import load
from variants.staged_reconstruction.train_static import setup, PIPE
from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
from variants.staged_reconstruction.stage2.model import compose
from variants.staged_reconstruction.joint_rgb.io import BASE, REPORT, RUNS, atomic_json, stopped, lock, freeze, verify
from variants.staged_reconstruction.joint_rgb.core import open_static_appearance


def main():
    assert not stopped() and not (REPORT/'native_check.json').exists()
    assert read(REPORT/'unit_tests.json')['passed']
    torch.set_num_threads(4); renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene, loss
    protocol = read(BASE/'protocol.json'); rows = []
    with lock('training.lock'):
        # Load all native dependencies, then freeze inputs/source before gradients.
        scene = JointScene(protocol, 'diffuse')
        freeze(); verify()
        for case in protocol['cases']:
            scene = JointScene(protocol, case)
            values = load(ROOT/protocol['synthetic_data']/case/'train')
            height, width = values['rgb'].shape[1:3]
            camera = calibrated_camera(values['views'][0], values['K'], width, height)
            target = torch.from_numpy(values['rgb'][0]).permute(2, 0, 1).cuda()
            with torch.no_grad():
                before = scene.render(renderer, camera, values['times'][0])
                direct = renderer(camera, compose(scene.static, [scene.at(values['times'][0])]), PIPE,
                    scene.center.new_zeros(3), 0., 0., 0.)
                assert torch.equal(before['render'], direct['render']) and torch.equal(before['alpha'], direct['alpha'])
                open_static_appearance(scene.static, True)
                after = scene.render(renderer, camera, values['times'][0])
                assert torch.equal(before['render'], after['render']) and torch.equal(before['alpha'], after['alpha'])
            optimizers = scene.optimizers(protocol)
            fixed = {name: getattr(scene.static, name).detach().clone() for name in ['_xyz', '_scaling', '_rotation']}
            gradient_rows = []
            for step in range(3):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                value = loss(scene.render(renderer, camera, values['times'][0]), target)
                assert value.isfinite(); value.backward()
                groups = {'static_opacity': [scene.static._opacity], 'static_SH_DC': [scene.static._features_dc],
                    'static_SH_rest': [scene.static._features_rest], 'dynamic_canonical': [scene.dynamic._xyz],
                    'grid': list(scene.grid.parameters()), 'network': list(scene.network.parameters())}
                norms = {}
                for name, parameters in groups.items():
                    grads = [p.grad for p in parameters if p.grad is not None]
                    assert grads and all(g.isfinite().all() for g in grads), name
                    norms[name] = sum(float(g.double().square().sum()) for g in grads)**.5
                if step == 2:
                    assert all(n > 0 for n in norms.values()), norms
                gradient_rows.append(norms)
                for optimizer in optimizers:
                    optimizer.step()
                scene.constrain()
            assert all(torch.equal(value, getattr(scene.static, name)) for name, value in fixed.items())
            rows.append({'case': case, 'same_value_RGB_alpha_exact': True, 'native_direct_PIDG_render_exact': True,
                'frozen_static_geometry_unchanged_after_three_updates': True,
                'finite_intended_gradients_nonzero_after_identity_head_updates': True, 'gradient_L2_by_smoke_step': gradient_rows})
        verify()
        atomic_json(REPORT/'native_check.json', {'passed': True, 'rows': rows,
            'formal_updates': 0, 'smoke_updates_per_case': 3,
            'source_manifest_sha256': sha(REPORT/'source_freeze.json'), 'source_sha256': sha(Path(__file__))})
        print('J1_NATIVE_FLAGS_OFF_GRADIENT_SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    main()
