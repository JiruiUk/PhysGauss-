"""Freeze initial geometry proposals and the original S3-U training budget."""
import numpy as np
import torch
from ..prepare_data import ROOT,REPORT,RUNS,read,write,sha
from ..early_depth.prepare import INITIAL,INITIAL_SHA
from .core import canonical_graph

EXP=REPORT/'local_rigidity'
WORK=RUNS/'local_rigidity'


def main():
    assert not (EXP/'protocol.json').exists(),'Preserve frozen protocols'
    assert sha(INITIAL)==INITIAL_SHA
    state=torch.load(INITIAL,map_location='cpu',weights_only=False)
    xyz=state['model']['pools.0._xyz'].numpy()
    edges,rest,weight=canonical_graph(xyz)
    WORK.mkdir(parents=True,exist_ok=True); graph=WORK/'graph.npz'
    assert not graph.exists()
    np.savez_compressed(graph,edges=edges,rest=rest,weight=weight)
    base=read(REPORT/'S3_union/protocol.json')
    screen=read(REPORT/'motion_integrity/protocol.json')
    plan=dict(arm='I5_local_rigidity',enabled_by_default=False,local_rigidity_enabled=True,relative_rotation_enabled=True,
              initial=INITIAL.relative_to(ROOT).as_posix(),initial_sha256=INITIAL_SHA,
              original_protocol_sha256=sha(REPORT/'S3_union/protocol.json'),
              graph=graph.relative_to(ROOT).as_posix(),graph_sha256=sha(graph),graph_k=20,graph_radius=.08,
              graph_edges=len(edges),graph_node_coverage=float(len(np.unique(edges))/len(xyz)),
              rigid_weight=.3,rotation_weight=.03,ramp_updates=400,edges_per_step=4096,
              warmup_updates=base['warmup_updates'],physics_updates=base['physics_updates'],
              actual_updates=base['warmup_updates']+base['physics_updates'],smoke_updates=200,
              fixed_cameras=screen['fixed_cameras'],screen_times=screen['screen_times'],
              full_acceptance_times=screen['full_acceptance_times'],acceptance=screen['acceptance'],
              original_physics_and_image_objective_retained=True,depth_or_tracks_used=False,
              no_object_count_or_material_segmentation=True,
              matched_control='Original S3-U initial state and16k+6k schedule; geometry/rotation prior is the only added training objective',
              shape_rest_policy='Learnable canonical vectors; frozen initial edge lengths only normalize errors. No parent-trained shape is treated as GT.',
              comparison_to_I1_is_not_single_switch=True)
    write(EXP/'protocol.json',plan)
    print('LOCAL_RIGIDITY_PREPARED',len(edges),plan['graph_node_coverage'],flush=True)


if __name__=='__main__': main()
