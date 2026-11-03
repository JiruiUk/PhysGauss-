"""Freeze an opt-in conditional invertible position experiment."""
from ..prepare_data import ROOT,REPORT,RUNS,read,write,sha
from ..early_depth.prepare import INITIAL,INITIAL_SHA

EXP=REPORT/'invertible_motion'
WORK=RUNS/'invertible_motion'


def main():
    assert not (EXP/'protocol.json').exists(),'Preserve experiment versions'
    assert sha(INITIAL)==INITIAL_SHA
    base=read(REPORT/'S3_union/protocol.json'); screen=read(REPORT/'motion_integrity/protocol.json')
    WORK.mkdir(parents=True,exist_ok=True)
    plan=dict(arm='I7_invertible_motion',enabled_by_default=False,invertible_motion_enabled=True,
        initial=INITIAL.relative_to(ROOT).as_posix(),initial_sha256=INITIAL_SHA,
        original_protocol_sha256=sha(REPORT/'S3_union/protocol.json'),
        initialization_seed=712,width=64,log_scale_bound=.2,translation_bound=.2,learning_rate=.0008,
        updated_axes=[0,1,2,0,1,2],spatial_bands=2,temporal_bands=4,
        original_covariance_heads_retained=True,original_position_head_inactive_when_enabled=True,
        warmup_updates=base['warmup_updates'],physics_updates=base['physics_updates'],
        actual_updates=base['warmup_updates']+base['physics_updates'],smoke_updates=200,
        fixed_cameras=screen['fixed_cameras'],screen_times=screen['screen_times'],
        full_acceptance_times=screen['full_acceptance_times'],acceptance=screen['acceptance'],
        original_physics_and_image_objective_retained=True,depth_or_tracks_used=False,
        no_object_count_or_material_segmentation=True,graph_losses_used=False,
        matched_control='Original S3-U initial full state and16k+6k schedule; new position parameterization and corresponding shared parameters',
        baseline_existing_parameters_and_Adam_state_preserved=True,new_optimizer_group='invertible_motion')
    write(EXP/'protocol.json',plan)
    print('INVERTIBLE_MOTION_PREPARED',flush=True)


if __name__=='__main__': main()
