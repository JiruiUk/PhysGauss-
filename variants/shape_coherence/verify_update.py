"""Actual isolated shape-only Adam update on a real immutable C1 snapshot."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from variants.c1_flow_ablation.results import motion_module
    m=motion_module()
    from variants.shape_coherence.model import local_shape
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    source=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
    os.environ['PIDG_METRIC_CONFIG']=str(source/'metric_config.json')
    g,d=m.load(source,'50000')
    state=torch.load(source/'checkpoints/resume.pt',map_location='cpu',weights_only=False)
    d.train_setting(SimpleNamespace(**state['opt']));d.optimizer.load_state_dict(state['deform_optimizer'])
    graph=torch.load(ROOT/'output/shape_coherence/audit/reference_graph.pt',map_location='cuda',weights_only=True)
    groups={'grid':list(d.grid4d.parameters()),'deform':list(d.deform.parameters())}
    before={k:[p.detach().clone() for p in v] for k,v in groups.items()}
    warp=warp_at(d,g.get_xyz.detach(),85/199)
    r=local_shape(graph['reference'],center_trajectory(g.get_xyz.detach(),warp['d_xyz']),
                  graph['anchors'][:128],graph['neighbors'][:128],graph['valid'][:128])
    loss=20*(1e-5*r['affine']+1e-6*r['collapse'])
    d.optimizer.zero_grad(set_to_none=True);loss.backward()
    gradients={k:sum(float(p.grad.norm()) for p in v if p.grad is not None) for k,v in groups.items()}
    assert all(torch.isfinite(p.grad).all() for v in groups.values() for p in v if p.grad is not None)
    assert all(p.grad is None for p in g.velocity_net.parameters()) and g.get_xyz.grad is None
    d.optimizer.step()
    changes={k:sum(float((p-b).abs().sum()) for p,b in zip(v,before[k])) for k,v in groups.items()}
    assert all(value>0 for value in changes.values()) and all(value>0 for value in gradients.values())
    result={'status':'passed','real_C1_full_checkpoint_optimizer_state':True,'loss':float(loss.detach()),
            'gradient_norm_sums':gradients,'actual_parameter_L1_changes':changes,
            'velocity_stress_and_canonical_xyz_receive_no_direct_shape_gradients':True,
            'new_parameters':0,'new_optimizer_groups':0}
    (ROOT/'experiments/shape_coherence/optimizer_update.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
