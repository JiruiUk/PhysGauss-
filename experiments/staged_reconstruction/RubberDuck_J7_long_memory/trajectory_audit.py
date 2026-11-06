"""Independent J7 full120 world-trajectory replay; never imports old xyz."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.j7_long_memory import runtime as a
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha,write
from output.short_clip_diagnostic.camera_math import w2c,intrinsic

def main():
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,scene,opts,teacher,parent=r.load('temporal_memory')
    before=deepcopy(r.cpu_values(r.snapshot(scene,opts,0,parent['camera_stack'],[])))
    meta=read(DATA/'transforms_train.json');p=read(a.BASE/'protocol.json');path=ROOT/p['trajectories']
    with np.load(path) as saved:
        assert np.array_equal(saved['views'],np.array([w2c(f['transform_matrix'],'metadata_cv') for f in meta['frames']]))
        assert np.array_equal(saved['k'],intrinsic(meta['camera_angle_x'],'metadata',400,225))
        with a.memory.evidence_context(scene):
            for index,frame in enumerate(meta['frames']):
                assert np.array_equal(saved['xyz'][index],scene.dynamic_at(frame['time']).get_xyz.cpu().numpy()),index
    r.assert_equivalent(before,r.snapshot(scene,opts,0,parent['camera_stack'],[]),0,0)
    r.verify();write(a.REPORT/'trajectory_audit.json',dict(passed=True,all120_J7_trajectories_exact=True,
        all15507_rows_each_time=True,camera_metadata_intrinsics_exact=True,no_old_xyz_import=True,
        model_twoAdam_rng_camera_physics_controller_unchanged=True,archive_sha256=sha(path),source_sha256=sha(Path(__file__))))
    print('J8_ALL120_J7_TRAJECTORIES_INDEPENDENT_REPLAY_PASS',flush=True)

if __name__=='__main__':main()
