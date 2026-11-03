"""One rendering policy shared by training, validation and full-state inference."""
import os
from pathlib import Path
import hashlib
import numpy as np
import torch
from .model import validate_rows, dynamic_rows


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


class StaticPool:
    def __init__(self, pc, config):
        path = Path(config['static_pool_binding'])
        if file_sha(path) != config['static_pool_binding_sha256']:
            raise ValueError('Static-pool binding checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            self.count = int(z['point_count'])
            if len(pc.get_xyz) != self.count:
                raise ValueError('Static-pool topology changed')
            if str(z['source_checkpoint_sha256']) != config['static_pool_source_sha256']:
                raise ValueError('Static-pool source mismatch')
            if not np.array_equal(z['particle_indices'], pc.particle_indices.detach().cpu().numpy()):
                raise ValueError('Static-pool particle mapping changed')
            ids = torch.tensor(z['static_ids'], device=pc.get_xyz.device, dtype=torch.long)
            validate_rows(ids, self.count)
            self.ids = ids
            self.reference = {}
            for key, width in [('d_xyz', 7), ('d_rotation', 4), ('d_scaling', 3)]:
                value = torch.as_tensor(z[key].copy(), device=pc.get_xyz.device, dtype=pc.get_xyz.dtype)
                if value.shape != (len(ids), width) or not bool(value.isfinite().all()):
                    raise ValueError('Invalid static reference: ' + key)
                self.reference[key] = value.detach()
        self.dynamic = dynamic_rows(ids, self.count)
        self.binding_sha256 = config['static_pool_binding_sha256']

    def apply(self, pc, d_xyz, d_rotation, d_scaling):
        if len(pc.get_xyz) != self.count:
            raise ValueError('Static pool cannot survive an unreceipted topology change')
        outputs = []
        for key, value in zip(('d_xyz', 'd_rotation', 'd_scaling'), (d_xyz, d_rotation, d_scaling)):
            if not torch.is_tensor(value):
                raise ValueError('Static-pool branch requires post-warmup tensor warps')
            outputs.append(value.index_copy(0, self.ids, self.reference[key]))
        return tuple(outputs)


def pool_for(pc, config=None):
    if config is None:
        # Same cached, immutable per-run configuration used by positive SH color.
        from variants.color_repair.color import policy
        config = policy()
    if not config.get('enabled', False) or not config.get('hard_static_pool', False):
        return None
    pool = getattr(pc, '_hard_static_pool', None)
    if pool is None:
        pool = StaticPool(pc, config)
        pc._hard_static_pool = pool
    elif pool.binding_sha256 != config['static_pool_binding_sha256']:
        raise ValueError('Static pool changed within a loaded model')
    return pool


def apply_warp(pc, d_xyz, d_rotation, d_scaling):
    pool = pool_for(pc)
    if pool is None:
        return d_xyz, d_rotation, d_scaling
    return pool.apply(pc, d_xyz, d_rotation, d_scaling)


def physics_candidates(pc):
    pool = pool_for(pc)
    return None if pool is None else pool.dynamic
