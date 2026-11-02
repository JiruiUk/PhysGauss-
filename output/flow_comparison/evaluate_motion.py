"""Post-selection motion diagnostic on unseen validation-source frame pairs.

Observed optical flow is estimated, not physical ground truth. Evaluation uses
raw source-to-target flow, without reconstruction-derived depth compensation.
No training, checkpoint selection or hyperparameter changes occur in this script.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'experiments/flow_comparison'
RUNS = ROOT/'runs/pidg_flow_comparison'
TRAIN = ROOT/'data/derived/kinematic_resplit/training'
TRAIN_CACHE = ROOT/'data/derived/flow_comparison_train_only'
CACHE = ROOT/'data/derived/flow_comparison_validation_eval'
TEACHER = ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'variants/flow_comparison/code'))
os.environ['PIDG_HASH_SOURCE_DIR'] = str(ROOT/'third_party/PIDG/hashencoder')
os.environ['PIDG_HASH_BUILD_DIR'] = str(ROOT/'third_party/PIDG/tmp_build')
from scene_PIDG.gaussian_model import GaussianModel
from scene_PIDG.deform_model import DeformModel
from scene_PIDG.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import focal2fov, fov2focal
from variants.flow_comparison.geometry import forward_backward_valid, sample, pixel_grid
from variants.flow_comparison.flow import next_projection, pixel_flow
from variants.flow_comparison import temporal
from variants.kinematic_tuning.model import center_trajectory

PIPE = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
BACKGROUND = torch.zeros(3, device='cuda')
META = read(TRAIN/'transforms_train.json')
FRAMES = {int(Path(f['file_path']).stem): f for f in META['frames']}
VAL = {int(Path(f['file_path']).stem): f for f in read(TRAIN/'transforms_test.json')['frames']}


def rgb(frame):
    with Image.open(TRAIN/(frame['file_path']+'.png')) as im:
        rgba = np.asarray(im.convert('RGBA'), dtype=np.float32)
    color = torch.from_numpy(rgba[..., :3]*rgba[..., 3:4]/255).permute(2, 0, 1)[None].cuda()
    return color, torch.from_numpy(rgba[..., 3]/255).cuda()


def camera(frame):
    fid = int(Path(frame['file_path']).stem)
    matrix = np.linalg.inv(np.array(frame['transform_matrix']))
    rotation = -matrix[:3, :3].T; rotation[:, 0] = -rotation[:, 0]
    fovx = META['camera_angle_x']; fovy = focal2fov(fov2focal(fovx, 1600), 900)
    return Camera(fid, rotation, -matrix[:3, 3], fovx, fovy, torch.zeros(3, 225, 400), None,
                  f'{fid:04d}', fid, fid=(fid-1)/199)


def load(directory, tag):
    args = runpy.run_path(str(directory/'model_config.py'))
    g = GaussianModel(3); g.load_ply(str(directory/'point_cloud'/('iteration_'+tag)/'point_cloud.ply'))
    material = torch.load(directory/'point_cloud'/('iteration_'+tag)/'material.pt', map_location='cuda', weights_only=True)
    g.velocity_net.load_state_dict(material['velocity_net'], strict=True)
    g.particle_indices = material['particle_indices'].to('cuda')
    g.force_sigma.data.copy_(material['force_sigma'].to('cuda'))
    if len(g.particle_indices) != len(g.get_xyz):
        raise ValueError('Snapshot material IDs do not match rendered geometry')
    d = DeformModel(args['grid_args'], args['network_args'], reg_spatial_able=False, reg_temporal_able=False)
    d.load_weights(str(directory), tag)
    return g, d


@torch.no_grad()
def targets():
    lock = read(RUNS/'selection_lock.json')
    if set(VAL)&set(FRAMES) or len(VAL) != 30:
        raise ValueError('Validation/train split overlap')
    pairs = [(a, min(b for b in FRAMES if b>a)) for a in sorted(VAL)]
    CACHE.mkdir(parents=True, exist_ok=True)
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in [TRAIN/'transforms_train.json', TRAIN/'transforms_test.json',
        REPORTS/'flow_model_lock.json', Path(__file__), *[TRAIN/(f['file_path']+'.png') for f in list(VAL.values())+list(FRAMES.values())]]}
    generation = {'inputs': inputs, 'selection_lock': lock, 'pairs': pairs,
        'purpose': 'post-selection validation motion diagnostic; never training or selection'}
    # JSON round-trip makes pair tuples comparable on reuse.
    generation = json.loads(json.dumps(generation))
    if (CACHE/'input_lock.json').exists():
        if read(CACHE/'input_lock.json') != generation:
            raise ValueError('Validation diagnostic input lock changed')
    else:
        (CACHE/'input_lock.json').write_text(json.dumps(generation, indent=2))
    if (CACHE/'manifest.json').exists():
        manifest = read(CACHE/'manifest.json')
        if any(sha(CACHE/p) != value for p, value in manifest['output_sha256'].items()):
            raise ValueError('Validation diagnostic cache changed')
        return manifest
    images = {}; masks = {}
    for a, b in pairs:
        color, alpha = rgb(VAL[a]); images[a] = F.interpolate(color, (448, 800), mode='bilinear', align_corners=False).cpu()
        masks[a] = F.interpolate(alpha[None, None], (225, 400), mode='bilinear', align_corners=False)[0, 0].cpu()
        if b not in images:
            color, alpha = rgb(FRAMES[b]); images[b] = F.interpolate(color, (448, 800), mode='bilinear', align_corners=False).cpu()
            masks[b] = F.interpolate(alpha[None, None], (225, 400), mode='bilinear', align_corners=False)[0, 0].cpu()
    flow_lock = read(REPORTS/'flow_model_lock.json'); sys.path.append(flow_lock['code_path'])
    from unimatch.unimatch import UniMatch
    model = UniMatch(num_scales=2, upsample_factor=4).cuda().eval().requires_grad_(False)
    weights = torch.load(flow_lock['weights_path'], map_location='cpu', weights_only=True)
    model.load_state_dict(weights.get('model', weights), strict=True)
    records = []
    for a, b in pairs:
        p = CACHE/f'{a:04d}_{b:04d}.npz'
        if not p.exists():
            flow = model(images[a].cuda(), images[b].cuda(), attn_type='swin', attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4], prop_radius_list=[-1, 1], pred_bidir_flow=True)['flow_preds'][-1]
            flow = F.interpolate(flow, (225, 400), mode='bilinear', align_corners=False)
            flow[:, 0] *= .5; flow[:, 1] *= 225/448
            forward, backward = flow.permute(0, 2, 3, 1)
            valid = forward_backward_valid(forward, backward)&(masks[a].cuda()>.95)&(sample(masks[b].cuda(), pixel_grid(225, 400, forward)+forward)>.99)
            assert valid.sum()>100
            np.savez_compressed(p, observed=forward.cpu().numpy(), valid=valid.cpu().numpy())
        records.append({'source': a, 'target': b, 'dt': (b-a)/199, 'file': p.name})
    del model; torch.cuda.empty_cache()
    manifest = {'generation': generation, 'pairs': records, 'output_sha256': {p.name: sha(p) for p in CACHE.glob('*.npz')}}
    (CACHE/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


@torch.no_grad()
def evaluate(directory, tag, manifest):
    os.environ['PIDG_METRIC_CONFIG'] = str(directory/'metric_config.json')
    config = read(directory/'metric_config.json'); source = config['velocity_source']
    g, d = load(directory, tag); rows = []
    for pair in manifest['pairs']:
        a, b, dt = pair['source'], pair['target'], pair['dt']; t = (a-1)/199
        cam = camera(VAL[a]); now = temporal.warp_at(d, g.get_xyz.detach(), t)
        package = render(cam, g, PIPE, BACKGROUND, now['d_xyz'], now['d_rotation'], now['d_scaling'], False,
            override_color=torch.zeros_like(g.get_xyz))
        later = temporal.warp_at(d, g.get_xyz.detach(), (b-1)/199)
        target_camera = camera(FRAMES[b])
        geometric, seen = pixel_flow(package, *next_projection(g, later, target_camera))
        v = temporal.velocity(g, d, g.get_xyz.detach(), g.particle_indices, t, source)
        current_centers = center_trajectory(g.get_xyz, now['d_xyz'])
        centers = current_centers+dt*v
        advected, _ = pixel_flow(package, *next_projection(g, later, target_camera, centers))
        camera_only, _ = pixel_flow(package, *next_projection(g, now, target_camera, current_centers))
        with np.load(CACHE/pair['file']) as data:
            target = torch.from_numpy(data['observed'].copy()).cuda(); valid = torch.from_numpy(data['valid'].copy()).cuda()
        # Fixed observation mask: missing model coverage cannot lower the score
        # by deleting the corresponding pixels from evaluation.
        row = {'source': a, 'target': b, 'dt': dt, 'valid_pixels': int(valid.sum()),
            'model_coverage': float(seen[valid].float().mean()),
            'zero_flow_EPE_px': float(target[valid].norm(dim=-1).mean())}
        for key, prediction in [('geometric', geometric), ('velocity', advected), ('camera_only_static', camera_only)]:
            assert prediction.isfinite().all()
            error = prediction[valid]-target[valid]
            row[key+'_EPE_px'] = float(error.norm(dim=-1).mean())
            row[key+'_normalized_L1'] = float((error/error.new_tensor([400, 225])).abs().mean())
        rows.append(row)
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('source', 'target', 'dt', 'valid_pixels')}
    probe_ids = torch.arange(256, device='cuda')*len(g.get_xyz)//256
    x = g.get_xyz.detach()[probe_ids]; material_ids = g.particle_indices[probe_ids]; sensitivity = []
    for a in sorted(VAL)[::6]:
        t = (a-1)/199; f = lambda s: temporal.position(d, x, s)
        v = temporal.derivative(f, t, .001)
        acceleration_function = f if source == 'trajectory' else lambda s: temporal.velocity(g, d, x, material_ids, s, source)
        order = 2 if source == 'trajectory' else 1
        acceleration = temporal.derivative(acceleration_function, t, 1/199, order)
        row = {'time': t, 'trajectory_speed_rms': float(v.square().mean().sqrt()),
               'acceleration_rms': float(acceleration.square().mean().sqrt())}
        for ratio in [.5, 2.]:
            other_v = temporal.derivative(f, t, .001*ratio)
            other_a = temporal.derivative(acceleration_function, t, ratio/199, order)
            row['velocity_step_ratio_'+str(ratio)] = float((other_v-v).square().mean().sqrt()/v.square().mean().sqrt().clamp_min(1e-8))
            row['acceleration_step_ratio_'+str(ratio)] = float((other_a-acceleration).square().mean().sqrt()/acceleration.square().mean().sqrt().clamp_min(1e-8))
        sensitivity.append(row)
    del g, d; torch.cuda.empty_cache()
    return {'model': directory.parent.name, 'snapshot': tag, 'velocity_source': source, 'metrics': metrics,
            'per_pair': rows, 'finite_difference_sensitivity': sensitivity}


if __name__ == '__main__':
    manifest = targets(); results = []
    for name in ['K1_flow_independent', 'K2_flow_trajectory']:
        for tag in ['best', '50000']:
            row = evaluate(RUNS/name/'rubber_duck', tag, manifest); results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'per_pair'}), flush=True)
    (REPORTS/'motion_validation.json').write_text(json.dumps({'status': 'complete', 'results': results,
        'protocol': '30 validation-source -> next training frame pairs; post-selection raw UniMatch optical flow; no estimated-depth compensation; not 3D ground truth',
        'input_lock_sha256': sha(CACHE/'input_lock.json'), 'manifest_sha256': sha(CACHE/'manifest.json')}, indent=2))
