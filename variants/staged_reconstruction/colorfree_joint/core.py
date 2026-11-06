"""Bank validation and explicit replacement; default-off leaves the caller intact."""
import torch

PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')


def validate_payload(payload):
    required = set(PARAMETERS) | {'initial_xyz', 'point_ids', 'born_at'}
    if set(payload) != required:
        raise ValueError('Unexpected static-bank schema')
    n = len(payload['_xyz'])
    shapes = {'_xyz': (n, 3), '_features_dc': (n, 1, 3), '_features_rest': (n, 15, 3),
              '_opacity': (n, 1), '_scaling': (n, 3), '_rotation': (n, 4),
              'initial_xyz': (n, 3), 'point_ids': (n,), 'born_at': (n,)}
    if not n:
        raise ValueError('Empty bank')
    for key, shape in shapes.items():
        if tuple(payload[key].shape) != shape or not torch.isfinite(payload[key]).all():
            raise ValueError('Invalid shape or nonfinite '+key)
    if len(torch.unique(payload['point_ids'])) != n:
        raise ValueError('Duplicate source row IDs')
    return n


def replace_static(scene, replacement=None, enabled=False):
    if not enabled:
        return scene
    if replacement is None:
        raise ValueError('Explicit source bank required')
    validate_payload(replacement.state_dict())
    scene.static = replacement
    return scene
