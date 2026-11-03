"""Read-only immediate exact reload and repeated native-gradient diagnostic."""
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.joint_rgb.io import BASE, REPORT, RUNS, atomic_json, verify, lock, stopped
from variants.staged_reconstruction.joint_rgb.core import assert_equivalent, rng_state, restore_rng
from variants.staged_reconstruction.synthetic_audit.input import load
from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
from variants.staged_reconstruction.train_static import setup


def differences(a, b, protocol):
    rows = {}
    for name, x in a.items():
        y = b[name]; delta = (x-y).abs()
        rows[name] = {'exact': bool(torch.equal(x, y)), 'max_abs': float(delta.max()),
            'above_original_tolerance': int((delta > protocol['recovery_atol']+protocol['recovery_rtol']*y.abs()).sum()),
            'elements': x.numel()}
    return rows


def main():
    assert not stopped() and not (REPORT/'recovery_diagnostic.json').exists()
    verify(); torch.set_num_threads(4); renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene, loss
    protocol = read(BASE/'protocol.json'); case = 'diffuse'
    saved = RUNS/'preflight/split'/case/'frozen_static/split20.pt'
    state = torch.load(saved, map_location='cpu', weights_only=False)
    before_rng = rng_state()
    with lock('training.lock'):
        scenes = [JointScene(protocol, case) for _ in range(2)]
        optimizers = []
        for scene in scenes:
            opts = scene.optimizers(protocol); scene.load_state_dict(state['model'])
            for opt, payload in zip(opts, state['optimizers']):
                opt.load_state_dict(payload)
            assert_equivalent(state['model'], {k: v.cpu() for k, v in scene.state_dict().items()}, 0, 0)
            assert_equivalent(state['optimizers'], [
                torch.load(saved, map_location='cpu', weights_only=False)['optimizers'][i] for i in range(2)], 0, 0)
            for opt, expected in zip(opts, state['optimizers']):
                def cpu(x):
                    if isinstance(x, torch.Tensor): return x.cpu()
                    if isinstance(x, dict): return {k: cpu(v) for k, v in x.items()}
                    if isinstance(x, list): return [cpu(v) for v in x]
                    return x
                assert_equivalent(expected, cpu(opt.state_dict()), 0, 0)
            optimizers.append(opts)
        values = load(ROOT/protocol['synthetic_data']/case/'train'); index = state['camera_stack'][-1]
        image = values['rgb'][index]; camera = calibrated_camera(values['views'][index], values['K'], image.shape[1], image.shape[0])
        target = torch.from_numpy(image).permute(2, 0, 1).cuda(); gradients, rgb, losses = [], [], []
        for repeat in range(3):
            scene = scenes[repeat % 2]
            scene.zero_grad(set_to_none=True); restore_rng(state)
            package = scene.render(renderer, camera, values['times'][index])
            value = loss(package, target); value.backward(); losses.append(float(value))
            rgb.append(package['render'].detach().cpu())
            gradients.append({name: p.grad.detach().cpu().clone() for name, p in scene.named_parameters() if p.grad is not None})
        hash_gradients = []
        scene = scenes[0]; xyz = scene.dynamic._xyz[:128].detach().clone()
        for _ in range(3):
            scene.grid.zero_grad(set_to_none=True)
            spatial = scene.grid.encode_spatial(xyz)
            spatial.square().sum().backward()
            hash_gradients.append({'xyz_embeddings': scene.grid.xyz_encoding.embeddings.grad.detach().cpu().clone()})
        for scene in scenes:
            assert_equivalent(state['model'], {k: v.cpu() for k, v in scene.state_dict().items()}, 0, 0)
        restore_rng(before_rng)
        atomic_json(REPORT/'recovery_diagnostic.json', {'status': 'completed_read_only_diagnostic',
            'immediate_two_scene_complete_model_and_both_Adam_reload_exact': True,
            'fixed_step20_next_view': index, 'identical_state_loss_three_repeats': losses,
            'same_state_RGB_exact': [bool(torch.equal(rgb[0], x)) for x in rgb[1:]],
            'whole_native_gradient_repeats': [differences(gradients[0], g, protocol) for g in gradients[1:]],
            'isolated_original_hash_embedding_gradient_repeats': [differences(hash_gradients[0], g, protocol) for g in hash_gradients[1:]],
            'no_model_optimizer_sampler_or_RNG_update': True, 'failed_full40_gate_unchanged': True,
            'formal_training_admitted': False, 'source_sha256': sha(Path(__file__)), 'step20_sha256': sha(saved),
            'diagnostic_plan_sha256': sha(BASE/'recovery_diagnostic_plan.md')})
        print('J1_NATIVE_RECOVERY_DIAGNOSTIC_COMPLETE_NO_UPDATES', flush=True)


if __name__ == '__main__':
    main()
