"""Independent read-only native RGB/depth/PLY/H264 audit after full training."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from experiments.staged_reconstruction.S1_joint_rgb.accepted_recovery.launch import configure, BASE, REPORT, RUNS, ROOT
from experiments.staged_reconstruction.S1_nuisance.export_synthetic import encode, depth_alpha_panel
from variants.staged_reconstruction.prepare_data import read, sha
from variants.staged_reconstruction.train_static import setup, render_static, PIPE
from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
from variants.staged_reconstruction.nuisance_static.evaluate import panel


def rgb(package):
    return package['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def main():
    io = configure(); io.verify(); assert not io.stopped()
    assert not (REPORT/'exports.json').exists()
    evaluation = read(REPORT/'synthetic_evaluation.json'); protocol = read(BASE/'protocol.json')
    torch.set_num_threads(4); renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene
    from scene_PIDG.gaussian_model import GaussianModel
    from plyfile import PlyData
    io.OUT.mkdir(parents=True, exist_ok=True); plys, videos, images = {}, {}, {}
    with io.lock('export.lock'):
        for case in protocol['cases']:
            models = {}
            for arm in protocol['arms']:
                path = RUNS/'formal'/case/arm/'checkpoint.pt'
                assert sha(path) == evaluation['checkpoint_audits'][case+'/'+arm]['checkpoint_sha256']
                state = torch.load(path, map_location='cpu', weights_only=False)
                model = JointScene(protocol, case, arm == 'revisable_static'); model.load_state_dict(state['model'])
                models[arm] = model
                for name, bank in [('static', model.static), ('dynamic_canonical_local', model.dynamic)]:
                    native = GaussianModel(3); native.active_sh_degree = 3
                    for key, parameter in bank.named_parameters(): setattr(native, key, parameter)
                    native.dynamic_mask = torch.zeros(len(bank._xyz), dtype=torch.bool, device='cuda')
                    ply = io.OUT/f'{case}_{arm}_{name}_6000.ply'; assert not ply.exists()
                    native.save_ply(str(ply)); vertex = PlyData.read(ply)['vertex']
                    xyz = np.stack([vertex[n] for n in ['x', 'y', 'z']], 1)
                    assert np.array_equal(xyz, bank._xyz.cpu().numpy())
                    assert all(np.isfinite(vertex[n]).all() for n in vertex.data.dtype.names)
                    plys[case+'/'+arm+'/'+name] = {'path': ply.relative_to(ROOT).as_posix(), 'sha256': sha(ply),
                        'vertices': len(xyz), 'positions_exact_checkpoint': True, 'all_fields_finite': True,
                        'coordinates': 'static world' if name == 'static' else 'canonical local; add checkpoint center before identity world transform'}
            private = ROOT/protocol['synthetic_data']/case/'evaluation_only'; meta = read(private/'cameras.json')
            k = np.asarray(meta['K']); static_cache, cameras, fixed, depth_sheets = {}, {}, [], []
            for row in meta['frames']:
                if row['time_index'] != 0: continue
                camera = calibrated_camera(np.asarray(row['world_to_camera']), k, meta['width'], meta['height'])
                cameras[row['camera']] = camera
                static_cache[row['camera']] = {arm: render_static(renderer, camera, model.static) for arm, model in models.items()}
                if row['camera'] in [0, 8, 24]:
                    with np.load(private/'gt'/f"c{row['camera']:02d}_t0.npz") as gt:
                        depth_sheets.append(depth_alpha_panel(list(static_cache[row['camera']].items()), gt, f"{case} c{row['camera']}"))
            depth_path = io.OUT/f'{case}_static_expected_depth_alpha.png'; assert not depth_path.exists()
            Image.fromarray(np.concatenate(depth_sheets)).save(depth_path); images[depth_path.relative_to(ROOT).as_posix()] = sha(depth_path)
            for row in meta['frames']:
                if row['camera'] != protocol['fixed_novel_camera']: continue
                camera = cameras[row['camera']]
                with np.load(private/'gt'/f"c{row['camera']:02d}_t{row['time_index']}.npz") as gt:
                    observed = gt['observed_rgb'].copy()
                for arm, model in models.items():
                    static = static_cache[row['camera']][arm]
                    dynamic = renderer(camera, model.at(row['time']), PIPE, model.center.new_zeros(3), 0., 0., 0.)
                    combined = model.render(renderer, camera, row['time'])
                    fixed.append(panel([('Evaluator observed RGB', observed), ('static only', rgb(static)),
                        ('unified dynamic only', rgb(dynamic)), ('depth-sorted joint', rgb(combined))],
                        f"{case} {arm} synthetic c24 t{row['time_index']}"))
                    for label, package in [('static', static), ('dynamic', dynamic), ('combined', combined)]:
                        path = io.OUT/f"{case}_{arm}_c24_t{row['time_index']}_{label}.png"; assert not path.exists()
                        Image.fromarray((rgb(package)*255).round().astype('uint8')).save(path)
                        images[path.relative_to(ROOT).as_posix()] = sha(path)
            path = io.OUT/f'{case}_novel24_temporal_decomposition.png'; assert not path.exists()
            Image.fromarray(np.concatenate(fixed)).save(path); images[path.relative_to(ROOT).as_posix()] = sha(path)
            def frames():
                for row in meta['frames']:
                    if io.stopped(): raise RuntimeError('Stop during export; preserve partial file')
                    camera = cameras[row['camera']]; values = []
                    for arm, model in models.items():
                        values += [(arm+' static', rgb(static_cache[row['camera']][arm])),
                            (arm+' combined', rgb(model.render(renderer, camera, row['time'])))]
                    yield panel(values, f"{case} synthetic c{row['camera']} t{row['time_index']}")
            videos[case] = encode(io.OUT/f'{case}_matched_static_joint.mp4', frames())
            videos[case]['content'] = 'learned unified dynamic Gaussian; synthetic camera/time grid, not Rubber Duck trajectory or independent physical identity'
            videos[case]['all_160_frames_decoded'] = True
        io.verify()
        io.atomic_json(REPORT/'exports.json', {'plys': plys, 'videos': videos, 'images': images,
            'native_86_188_Rubber_Duck_outputs': False, 'manual_image_review_pending': True,
            'depth_contract': 'nativeD/alpha; black alpha<=.5 is missing, never zero error',
            'source_sha256': sha(Path(__file__))})
        print('J1_A_NATIVE_EXPORTS_AND_FULL_VIDEO_DECODE_COMPLETE_PENDING_MANUAL_REVIEW', flush=True)


if __name__ == '__main__': main()
