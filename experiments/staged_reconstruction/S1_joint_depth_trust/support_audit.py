"""Posthoc evaluator-only support replay; no criterion/threshold or training changes."""
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.prepare_data import ROOT, read, sha
from variants.staged_reconstruction.train_static import setup, render_static
from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
from experiments.staged_reconstruction.S1_joint_depth_trust.launch import configure, BASE, REPORT, RUNS


@torch.no_grad()
def main():
    io = configure(); io.verify(); assert not io.stopped()
    assert not (REPORT/'support_audit.json').exists()
    torch.set_num_threads(4); renderer = setup()
    from variants.staged_reconstruction.joint_rgb.runtime import JointScene
    protocol = read(BASE/'protocol.json'); evaluation = read(REPORT/'synthetic_evaluation.json')
    cases = {}
    for case in protocol['cases']:
        models = {}
        for arm in protocol['arms']:
            path = RUNS/'formal'/case/arm/'checkpoint.pt'
            assert sha(path) == evaluation['checkpoint_audits'][case+'/'+arm]['checkpoint_sha256']
            state = torch.load(path, map_location='cpu', weights_only=False)
            scene = JointScene(protocol, case, arm == 'revisable_static'); scene.load_state_dict(state['model'])
            scene.requires_grad_(False); models[arm] = scene
        private = ROOT/protocol['synthetic_data']/case/'evaluation_only'; meta = read(private/'cameras.json')
        rows = []
        for row in meta['frames']:
            if row['time_index'] != 0: continue
            assert not io.stopped()
            camera = calibrated_camera(np.asarray(row['world_to_camera']), np.asarray(meta['K']), meta['width'], meta['height'])
            with np.load(private/'gt'/f"c{row['camera']:02d}_t0.npz") as gt: depth_gt = gt['static_depth'].copy()
            surface, background = depth_gt > 0, depth_gt <= 0
            record = {'camera': row['camera'], 'novel': not row['seen_camera'], 'arms': {}}
            previous = next(r for r in evaluation['results'][case]['static_frames'] if r['camera'] == row['camera'])
            for arm, scene in models.items():
                p = render_static(renderer, camera, scene.static)
                alpha = p['alpha'].squeeze().cpu().numpy(); d = p['depth'].squeeze().cpu().numpy()/np.maximum(alpha, 1e-8)
                valid = (alpha > .5) & np.isfinite(d)
                accurate = valid & (np.abs(d-depth_gt) <= .05)
                expected = previous['models'][arm]
                assert np.count_nonzero(valid & surface) == expected['ray_counts']['static_valid_pixels']
                assert np.count_nonzero(accurate & surface) == expected['ray_counts']['accurate_static_ray_pixels']
                assert abs(float(alpha[background].mean())-expected['background_alpha']) <= 1e-6
                record['arms'][arm] = {'background_GT_pixels': int(background.sum()),
                    'outside_GT_alpha_gt05_pixels': int(np.count_nonzero(background & (alpha > .5))),
                    'surface_GT_pixels': int(surface.sum()), 'accurate_surface_pixels': int(np.count_nonzero(accurate & surface))}
            rows.append(record)
        aggregate = {}
        for subset in ['all32', 'novel8']:
            selection = rows if subset == 'all32' else [r for r in rows if r['novel']]
            aggregate[subset] = {arm: {key: sum(r['arms'][arm][key] for r in selection)
                for key in selection[0]['arms'][arm]} for arm in protocol['arms']}
        cases[case] = {'aggregate': aggregate, 'frames': rows, 'original_per_camera_alpha_and_ray_metrics_replayed': True}
        print('J3_SUPPORT_AUDIT', case, aggregate['novel8'], flush=True)
    io.verify(); io.atomic_json(REPORT/'support_audit.json', {'passed': True,
        'posthoc_evaluator_only_not_new_training_gate': True, 'existing_alpha05_depth005_thresholds_only': True,
        'cases': cases, 'GT_assets_or_masks_exported': False,
        'evaluation_sha256': sha(REPORT/'synthetic_evaluation.json'), 'source_sha256': sha(Path(__file__))})


if __name__ == '__main__': main()
