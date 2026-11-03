"""Atomic full-state snapshots including sampler and derived field graph state."""
import os
import random
import numpy as np
import torch
from ..prepare_data import read, write, sha
from ..state_recovery.runtime import graph_state, restore_graphs


def save(directory, scene, optimizer, step, stack, phase, protocol_hash, parent_hash):
    path = directory/'checkpoint.pt'
    temp = directory/'checkpoint.tmp'
    torch.save({'model': scene.state_dict(), 'optimizer': optimizer.state_dict(), 'iteration': step,
        'camera_stack': list(stack), 'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
        'physics_graphs': graph_state(scene), 'stage': 'S3-U', 'phase': phase,
        'protocol_sha256': protocol_hash, 'parent_sha256': parent_hash,
        'physical_body_number': None, 'instance_identity_available': False}, temp)
    os.replace(temp, path)
    write(directory/'checkpoint.json', {'iteration': step, 'sha256': sha(path), 'full_state': True, 'phase': phase})


def restore(path, scene, optimizer, protocol_hash):
    assert sha(path) == read(path.with_name('checkpoint.json'))['sha256']
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['protocol_sha256'] == protocol_hash
    scene.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    restore_graphs(scene, state['physics_graphs'])
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    torch.cuda.set_rng_state_all(state['rng_cuda'])
    return state
