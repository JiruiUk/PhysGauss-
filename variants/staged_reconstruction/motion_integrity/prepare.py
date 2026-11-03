"""Read-only real-parent audit and train-only graph preparation."""
import argparse
import numpy as np
import torch
from ..prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha
from ..train_static import setup
from ..union.prepare import verify_sources
from ..union.runtime import restore
from .core import trajectory_graph

EXP = REPORT/'motion_integrity'
WORK = RUNS/'motion_integrity'
PARENT = RUNS/'S3_union/physics/formal/checkpoint.pt'
PARENT_SHA = '8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0'


def load_parent():
    verify_sources()
    assert sha(PARENT) == PARENT_SHA
    renderer = setup()
    from ..union.model import UnionScene
    protocol = read(REPORT/'S3_union/protocol.json')
    scene = UnionScene(protocol)
    opt = scene.optimizer()
    state = restore(PARENT, scene, opt, sha(REPORT/'S3_union/protocol.json'))
    assert state['iteration'] == 6000 and state['phase'] == 'physics'
    return renderer, scene, opt, state, protocol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--enable-motion-integrity', action='store_true')
    args = p.parse_args()
    if not args.enable_motion_integrity:
        raise SystemExit('Motion integrity is opt-in; pass --enable-motion-integrity')
    torch.set_num_threads(4)
    EXP.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    assert not (EXP/'protocol.json').exists(), 'Keep prepared protocol immutable'
    _, scene, _, state, protocol = load_parent()
    frames = read(DATA/'transforms_train.json')['frames']
    selected = np.linspace(0, len(frames)-1, 40).round().astype(int)
    with torch.no_grad():
        positions = np.stack([scene.dynamic_at(frames[i]['time']).get_xyz.cpu().numpy() for i in selected])
        scales = np.stack([scene.dynamic_at(frames[i]['time']).get_scaling.cpu().numpy() for i in selected])
    graph = trajectory_graph(positions)
    np.savez_compressed(WORK/'graph.npz', **graph, source_indices=selected,
                        source_times=np.array([frames[i]['time'] for i in selected]), parent_positions=positions)
    degree = np.bincount(graph['edges'].ravel(), minlength=positions.shape[1])
    fractions = []
    for frame in positions:
        distance = np.linalg.norm(frame[graph['edges'][:, 1]]-frame[graph['edges'][:, 0]], axis=-1)
        fractions.append(float(np.mean(np.abs(np.log(np.maximum(distance, 1e-7)/graph['rest'])) > np.log1p(.2))))
    report = {'parent_sha256': PARENT_SHA, 'parent_iteration': state['iteration'],
              'optimizer_updates': 0, 'population': positions.shape[1], 'edges': len(graph['edges']),
              'nodes_with_edges': int((degree > 0).sum()), 'node_coverage': float((degree > 0).mean()),
              'degree_quantiles': np.quantile(degree, [0, .1, .5, .9, 1]).tolist(),
              'curve_distance_quantiles': np.quantile(graph['curve_max'], [.1, .5, .9, 1]).tolist() if len(graph['edges']) else [],
              'initial_violation_per_time': fractions, 'scale_quantiles': np.quantile(scales, [0, .1, .5, .9, 1]).tolist(),
              'training_metadata_sha256': sha(DATA/'transforms_train.json'),
              'source_frames': [frames[i]['file_path'] for i in selected],
              'identity_or_depth_GT': False, 'graph_is_frozen_model_prior': True,
              'limit': 'Trajectory proximity cannot certify same-object membership; parent errors can persist.'}
    write(EXP/'parent_graph_audit.json', report)
    plan = {'enabled_by_default': False, 'parent': PARENT.relative_to(ROOT).as_posix(), 'parent_sha256': PARENT_SHA,
            'graph': (WORK/'graph.npz').relative_to(ROOT).as_posix(), 'graph_sha256': sha(WORK/'graph.npz'),
            'arms': ['I0_control', 'I1_curve_isometry'], 'actual_updates': 2000, 'width': 1600,
            'strain_weight': .03, 'ramp_updates': 400, 'strain_every': 1, 'edges_per_update': 4096,
            'extra_time_policy': 'deterministic coprime train-time schedule, no RNG consumption',
            'parent_optimizer_RNG_sampler_retained': True, 'original_physics_retained': True,
            'graph_k': 12, 'graph_max_radius': .08, 'strain_log_tolerance': float(np.log1p(.2)),
            'rest_policy': 'Frozen median parent edge lengths over 40 real training times',
            'fixed_cameras': [2, 43, 86, 146, 188, 200],
            'screen_times': [2, 27, 43, 62, 86, 108, 146, 172, 200],
            'full_acceptance_times': list(range(1, 201)),
            'acceptance': {'validation_dynamic_PSNR_drop_max_dB': .3,
                           'validation_dynamic_alpha_min': .95,
                           'native_visual_no_obvious_missing_parts_or_detached_fragments': True,
                           'all_fixed_camera_full_sequences_required': True,
                           'proxy_metrics_alone_cannot_admit': True},
            'test_policy': 'No test RGB during preparation/training or iterative selection; final selected model only',
            'no_instance_count_no_material_segmentation': True}
    write(EXP/'protocol.json', plan)
    frozen = {f.relative_to(ROOT).as_posix(): sha(f) for f in [PARENT, REPORT/'S3_union/protocol.json', DATA/'transforms_train.json']}
    for f in (ROOT/'variants/staged_reconstruction/motion_integrity').glob('*.py'):
        frozen[f.relative_to(ROOT).as_posix()] = sha(f)
    write(EXP/'preparation_freeze.json', frozen)
    print('MOTION_GRAPH_PREPARED', report, flush=True)


if __name__ == '__main__':
    main()
