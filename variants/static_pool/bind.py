"""Capture the accepted rows' own reference warps from the immutable common15k."""
import sys
import numpy as np
import torch
from .run import ROOT, REPORT, RUNS, source, plan, engine
from .prepare import prepare


def main():
    torch.set_num_threads(4); repo = prepare(); sys.path.insert(0, str(repo))
    from variants.association_training.evaluate import load
    from variants.flow_comparison.temporal import warp_at
    from variants.static_pool.model import validate_rows
    path = RUNS/'static_reference.npz'
    if (REPORT/'binding.json').exists():
        b = engine.read_json(REPORT/'binding.json')
        assert engine.sha256(path) == b['sha256']
        assert b['source_checkpoint_sha256'] == plan()['source_checkpoint_sha256']
        return
    old = engine.read_json(ROOT/plan()['binding_metadata'])
    assert old['source_checkpoint_sha256'] == plan()['source_checkpoint_sha256']
    assert engine.sha256(old['path']) == old['sha256']
    g, d, proof = load(source())
    assert proof['sha256'] == plan()['source_checkpoint_sha256'] and proof['iteration'] == 15000
    with np.load(old['path'], allow_pickle=False) as z:
        ids, times = z['static_ids'].copy(), z['static_reference_frames'].copy()
        assert len(ids) == old['static_points'] == 14624
    validate_rows(torch.tensor(ids, dtype=torch.long), len(g.get_xyz))
    arrays = {'static_ids': ids, 'reference_frames': times, 'point_count': np.asarray(len(g.get_xyz)),
        'source_checkpoint_sha256': np.asarray(proof['sha256']),
        'particle_indices': g.particle_indices.cpu().numpy()}
    for k, width in [('d_xyz',7), ('d_rotation',4), ('d_scaling',3)]:
        arrays[k] = np.empty((len(ids),width), np.float32)
    with torch.no_grad():
        for n in np.unique(times):
            selected = np.flatnonzero(times == n)
            w = warp_at(d, g.get_xyz.detach(), (int(n)-1)/199)
            rows = torch.tensor(ids[selected], device='cuda')
            for k in ['d_xyz','d_rotation','d_scaling']:
                arrays[k][selected] = w[k][rows].cpu().numpy()
    assert all(np.isfinite(arrays[k]).all() for k in ['d_xyz','d_rotation','d_scaling'])
    assert np.linalg.norm(arrays['d_xyz'][:,:4],axis=1).min()>1e-8
    RUNS.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
    assert engine.sha256(source()/'checkpoints/resume.pt') == proof['sha256']
    engine.write_json(REPORT/'binding.json', {'path': str(path), 'sha256': engine.sha256(path),
        'source_checkpoint_sha256': proof['sha256'], 'source_row_binding_sha256': old['sha256'],
        'point_count': len(g.get_xyz), 'static_points': len(ids), 'dynamic_or_unknown_points': len(g.get_xyz)-len(ids),
        'static_blue_contribution_mean': old['static_mean_blue_alpha_share'],
        'static_blue_contribution_time_thirds': old['static_blue_share_time_thirds'],
        'train_only': True, 'new_Gaussians': 0, 'new_learnable_parameters': 0,
        'reference_policy': plan()['static_reference'], 'physical_material_ID': False})
    print('Bound static references:',len(ids),'of',len(g.get_xyz),flush=True)


if __name__ == '__main__': main()
