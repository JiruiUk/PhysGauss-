"""Complete end-of-iteration PIDG state; only load checkpoints created locally."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import torch
import numpy as np

PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity', 'force_sigma')
BUFFERS = ('max_radii2D', 'xyz_gradient_accum', 'denom', 'particle_indices', 'dynamic_mask')


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_training_state(local, destination):
    g, d, scene = local['gaussians'], local['deform'], local['scene']
    torch.cuda.synchronize()
    cameras = scene.getTrainCameras()
    camera_indices = {id(camera): i for i, camera in enumerate(cameras)}
    state = {
        'format_version': 1, 'boundary': 'after_optimizer_step_and_lr_update',
        'iteration': int(local['iteration']),
        'gaussian_parameters': cpu_tree({name: getattr(g, name) for name in PARAMETERS}),
        'gaussian_buffers': cpu_tree({name: getattr(g, name) for name in BUFFERS}),
        'gaussian_scalars': {name: getattr(g, name) for name in ('active_sh_degree', 'max_sh_degree', 'spatial_lr_scale', 'percent_dense')},
        'velocity_net': cpu_tree(g.velocity_net.state_dict()),
        'gaussian_optimizer': cpu_tree(g.optimizer.state_dict()),
        'grid4d': cpu_tree(d.grid4d.state_dict()),
        'deform': cpu_tree(d.deform.state_dict()),
        'deform_optimizer': cpu_tree(d.optimizer.state_dict()),
        'dataset': vars(local['dataset']).copy(), 'opt': vars(local['opt']).copy(), 'pipe': vars(local['pipe']).copy(),
        'train_camera_names': [c.image_name for c in cameras],
        'test_camera_names': [c.image_name for c in scene.getTestCameras()],
        'viewpoint_stack': None if local['viewpoint_stack'] is None else [camera_indices[id(c)] for c in local['viewpoint_stack']],
        'ema_loss_for_log': float(local['ema_loss_for_log']), 'best_psnr': float(local['best_psnr']),
        'best_iteration': int(local['best_iteration']),
        'testing_iterations': local['testing_iterations'], 'saving_iterations': local['saving_iterations'],
        'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(), 'rng_cuda': cpu_tree(torch.cuda.get_rng_state_all()),
        'torch_version': str(torch.__version__), 'created_unix': time.time(),
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    with destination.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    manifest = {'path': str(destination), 'iteration': state['iteration'], 'sha256': digest,
                'size_bytes': destination.stat().st_size, 'format_version': 1,
                'optimizer_saved': True, 'rng_saved': True, 'created_unix': time.time()}
    destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('FULL_CHECKPOINT_SAVED ' + json.dumps(manifest), flush=True)
    return manifest


def load_state(path):
    # Local trusted training checkpoint includes Python/NumPy random states.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['format_version'] != 1 or state['boundary'] != 'after_optimizer_step_and_lr_update':
        raise ValueError('Unsupported checkpoint boundary or version')
    return state


def restore_models(state, gaussians, deform, opt):
    for name, tensor in state['gaussian_parameters'].items():
        setattr(gaussians, name, torch.nn.Parameter(tensor.to('cuda').requires_grad_(True)))
    gaussians.velocity_net.load_state_dict(state['velocity_net'])
    gaussians.training_setup(opt)
    for name, tensor in state['gaussian_buffers'].items():
        setattr(gaussians, name, tensor.to('cuda'))
    for name, value in state['gaussian_scalars'].items():
        setattr(gaussians, name, value)
    gaussians.optimizer.load_state_dict(state['gaussian_optimizer'])
    deform.grid4d.load_state_dict(state['grid4d'])
    deform.deform.load_state_dict(state['deform'])
    deform.optimizer.load_state_dict(state['deform_optimizer'])


def restore_training_state(path, gaussians, deform, opt, scene):
    state = load_state(path)
    if vars(opt) != state['opt']:
        raise ValueError('Training options differ from the saved checkpoint')
    restore_models(state, gaussians, deform, opt)
    def reorder(cameras, names):
        by_name = {camera.image_name: camera for camera in cameras}
        if len(by_name) != len(cameras) or set(by_name) != set(names):
            raise ValueError('Camera set differs from saved checkpoint')
        cameras[:] = [by_name[name] for name in names]
    cameras = scene.getTrainCameras()
    reorder(cameras, state['train_camera_names'])
    reorder(scene.getTestCameras(), state['test_camera_names'])
    state['viewpoint_stack'] = None if state['viewpoint_stack'] is None else [cameras[i] for i in state['viewpoint_stack']]
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    torch.cuda.set_rng_state_all(state['rng_cuda'])
    return state


def install_live_export(destination):
    """Called by the local debugger; save once on the training thread at a safe boundary."""
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    frames = list(sys._current_frames().values())
    training = None
    for frame in frames:
        while frame is not None:
            if frame.f_code.co_name == 'training' and frame.f_code.co_filename.endswith('train_pidg.py'):
                training = frame
                break
            frame = frame.f_back
        if training is not None:
            break
    if training is None:
        raise RuntimeError('Live training frame not found')
    deform = training.f_locals['deform']
    original = deform.update_learning_rate
    if getattr(deform, '_full_checkpoint_export_installed', False):
        Path(destination).with_suffix('.armed.json').write_text(json.dumps({'pid': os.getpid(), 'armed': True}), encoding='utf-8')
        return
    def export_after_step(iteration):
        original(iteration)
        try:
            caller = sys._getframe(1)
            if caller.f_code.co_name != 'training':
                raise RuntimeError('Checkpoint called outside training loop')
            save_training_state(caller.f_locals, destination)
        except Exception:
            Path(destination).with_suffix('.error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        # Keep the live state until an external reader verifies the disk checkpoint.
        while True:
            time.sleep(1)
    deform.update_learning_rate = export_after_step
    deform._full_checkpoint_export_installed = True
    Path(destination).with_suffix('.armed.json').write_text(json.dumps({'pid': os.getpid(), 'armed': True}), encoding='utf-8')
