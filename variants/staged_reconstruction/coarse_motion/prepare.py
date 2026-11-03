"""Freeze a capacity prior and the original initial state/training schedule."""
import numpy as np
import torch
from ..prepare_data import ROOT,REPORT,RUNS,read,write,sha
from ..early_depth.prepare import INITIAL,INITIAL_SHA
from .core import make_controls

EXP=REPORT/'coarse_motion'
WORK=RUNS/'coarse_motion'


def main():
    assert not (EXP/'protocol.json').exists(),'Preserve experiment versions'
    assert sha(INITIAL)==INITIAL_SHA
    state=torch.load(INITIAL,map_location='cpu',weights_only=False)
    rows,nodes,radius,nearest=make_controls(state['model']['pools.0._xyz'].numpy(),16)
    WORK.mkdir(parents=True,exist_ok=True); path=WORK/'controls.npz'; assert not path.exists()
    np.savez_compressed(path,rows=rows,xyz=nodes,radius=radius)
    base=read(REPORT/'S3_union/protocol.json'); screen=read(REPORT/'motion_integrity/protocol.json')
    plan=dict(arm='I6_coarse_motion',enabled_by_default=False,shared_motion_enabled=True,
        initial=INITIAL.relative_to(ROOT).as_posix(),initial_sha256=INITIAL_SHA,
        original_protocol_sha256=sha(REPORT/'S3_union/protocol.json'),
        controls=path.relative_to(ROOT).as_posix(),controls_sha256=sha(path),control_count=16,control_knn=4,
        fixed_radius=float(radius[0]),radius_policy='half of initial nearest-control distance90th percentile, uniform across controls',
        nearest_control_distance_quantiles={str(q):float(np.quantile(nearest,q)) for q in [.5,.9,.99,1.]},
        full_sparse_from_update=1,mix=1.,no_new_trainable_parameters=True,
        warmup_updates=base['warmup_updates'],physics_updates=base['physics_updates'],
        actual_updates=base['warmup_updates']+base['physics_updates'],smoke_updates=200,
        fixed_cameras=screen['fixed_cameras'],screen_times=screen['screen_times'],
        full_acceptance_times=screen['full_acceptance_times'],acceptance=screen['acceptance'],
        original_physics_and_image_objective_retained=True,depth_or_tracks_used=False,
        no_object_count_or_material_segmentation=True,graph_losses_used=False,
        matched_control='Original S3-U initial full state, identical16k+6k schedule; only shared motion representation changes',
        comparison_to_I2_is_not_single_switch=True)
    write(EXP/'protocol.json',plan)
    print('COARSE_MOTION_PREPARED',len(nodes),float(radius[0]),flush=True)


if __name__=='__main__': main()
