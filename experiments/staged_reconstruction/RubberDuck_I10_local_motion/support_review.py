"""Independent no-extrapolation/support and actual parameter-update audit."""
from pathlib import Path
import numpy as np
import torch
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha

B=Path(__file__).parent;V=B/'v1';R=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_I10_local_motion/v1'


def main():
    torch.set_num_threads(4);p=read(B/'protocol.json');evidence=read(ROOT/p['evidence'])
    manifest=read(V/'support.json');obs={};pairs={}
    with np.load(ROOT/p['trajectories']) as f:xyz=f['xyz']
    for track in evidence['rows']:
        if not track['accepted']:continue
        anchor=track['binding']['row']
        visible=track['visibility_by_row'][str(anchor)]
        for local in range(len(track['uv'])):
            if not visible[local]['passed']:continue
            frame=track['start']+local;obs.setdefault((frame,anchor),[]).append(track['uv'][local])
            if track['neighborhood']['qualified']:
                for neighbor in track['neighborhood']['rows']:
                    i,j=sorted([anchor,neighbor]);pairs[(frame,i,j)]=(xyz[frame,j]-xyz[frame,i]).tolist()
    assert len(obs)==manifest['observations']==1325 and len(pairs)==manifest['pairs']==9367
    assert len({k[0] for k in obs})==89 and len({k[0] for k in pairs})==71
    for frame in range(120):
        assert sum(key[0]==frame for key in obs)==manifest['track_counts_by_frame'][str(frame)]
        assert sum(key[0]==frame for key in pairs)==manifest['pair_counts_by_frame'][str(frame)]
    audit={};last={}
    for arm in p['arms']:
        directory=R/'formal'/arm
        initial=torch.load(directory/'initial.pt',map_location='cpu',weights_only=False)
        final=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert final['iteration']==2000 and final['stage']=='L1_I10_supported_local_motion'
        assert final['original_global_coverage_not_promoted'] and final['local_prior_is_parent_conditioned']
        changes={}
        for prefix in ['pools.0._xyz','grids.','networks.']:
            values=[(final['model'][n]-v).abs().sum().item() for n,v in initial['model'].items() if n.startswith(prefix)]
            assert values and np.isfinite(values).all() and sum(values)>0
            changes[prefix]=sum(values)
        trace=final['sample_trace']
        audit[arm]=dict(steps=2000,parameter_absolute_changes=changes,
            RGB_samples_with_track_support=sum(manifest['track_counts_by_frame'][str(i)]>0 for i in trace),
            RGB_samples_with_local_support=sum(manifest['pair_counts_by_frame'][str(i)]>0 for i in trace),
            actual_local_loss_enabled=arm=='local_motion',complete_checkpoint_sha256=sha(directory/'checkpoint.pt'))
        last[arm]=final['model']
    assert any(not torch.equal(last[p['arms'][0]][n],last[p['arms'][1]][n]) for n in last[p['arms'][0]])
    old=read(ROOT/'experiments/staged_reconstruction/RubberDuck_I10_track_neighborhood/visible/v1/final_review.json')
    assert old['status']=='completed_not_admitted' and not old['neighborhood_gate_passed'] and old['optimizer_updates']==0
    write(V/'support_audit.json',dict(passed=True,all_support_equations_and_frame_denominators_recomputed=True,
        no_inferred_edges_or_targets_outside_supported_frames=True,old_global_gate_failure_preserved=True,
        actual_two_arm_parameter_updates=True,arms=audit,source_sha256=sha(Path(__file__))))
    print('L1_SUPPORT_AND_ACTUAL_UPDATE_AUDIT_PASS',audit,flush=True)


if __name__=='__main__':main()
