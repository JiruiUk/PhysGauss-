"""Minimal independent fork; original PIDG/C1 sources are never edited."""
from pathlib import Path
import shutil
from variants.color_repair.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

HERE = Path(__file__).parent


def prepare():
    source = parent_prepare(); parent = runtime_hashes(source)
    target, receipt = HERE/'code', HERE/'source_manifest.json'
    if target.exists():
        if read_json(receipt) != {'parent': parent, 'fork': runtime_hashes(target)}:
            raise ValueError('Static-pool fork/source receipt changed')
        return target
    for name in parent:
        p = target/name; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, p)
    p = target/'train_pidg.py'; text = p.read_text(encoding='utf-8')
    text = replace_once(text, 'from variants.color_repair.runtime import Tuning',
                        'from variants.static_pool.runtime import Tuning')
    old = '''    N = coords.shape[0]
    # If N is smaller than the sample size, use all particles directly
    M = min(N, sample_size)
    # Randomly select M indices
    perm = torch.randperm(N, device=coords.device)[:M]
'''
    new = '''    from variants.static_pool.render import physics_candidates
    eligible = physics_candidates(gaussians)
    if eligible is None:
        N = coords.shape[0]
        M = min(N, sample_size)
        perm = torch.randperm(N, device=coords.device)[:M]
    else:
        N = len(eligible)
        if N == 0:
            return coords.sum() * 0.0
        M = min(N, sample_size)
        perm = eligible[torch.randperm(N, device=coords.device)[:M]]
'''
    p.write_text(replace_once(text, old, new), encoding='utf-8')
    p = target/'gaussian_renderer/__init__.py'; text = p.read_text(encoding='utf-8')
    text = replace_once(text, '    # Create zero tensor.',
        '    from variants.static_pool.render import apply_warp\n'
        '    d_xyz, d_rotation, d_scaling = apply_warp(pc, d_xyz, d_rotation, d_scaling)\n\n'
        '    # Create zero tensor.')
    p.write_text(text, encoding='utf-8')
    assert runtime_hashes(source) == parent
    write_json(receipt, {'parent': parent, 'fork': runtime_hashes(target)})
    return target
