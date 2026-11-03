"""Bind train-only image fragments once to a reviewed, immutable S3 state."""
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from ..prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha
from ..train_static import setup,camera,PIPE
from ..recovery.coverage import project


def eligible_rows(error,depth_valid,component_mass,total_mass,patch_mass,source=False):
    """Observation support only; these tests never certify material identity."""
    return (depth_valid & torch.isfinite(error) & (error <= (1.5 if source else 2.))
        & (component_mass >= .02) & (component_mass >= .95*total_mass)
        & (patch_mass >= .005))


def conflicts(a,b,tolerance=.75):
    if (a['local_early_object'],a['local_row']) != (b['local_early_object'],b['local_row']):return False
    targets=dict(zip(a['frames'],a['uv_400']))
    return any(n in targets and np.linalg.norm(np.asarray(uv)-targets[n])>tolerance
        for n,uv in zip(b['frames'],b['uv_400']))


def load_physics_state(path):
    from ..stage3.train import PhysicsScene,REC_RUNS
    bootstrap=torch.load(REC_RUNS/'S2_objects/checkpoint.pt',map_location='cpu',weights_only=False)
    scene=PhysicsScene(bootstrap)
    from ..state_recovery.physics import CurrentWorldField
    scene.physics=CurrentWorldField(len(scene.objects))
    state=torch.load(path,map_location='cpu',weights_only=False)
    scene.load_state_dict(state['model'])
    return scene,state,bootstrap


def visible_masses(renderer,scene,cam,t,mask,uv):
    # Three channels measure complete alpha*T integrals for component, whole
    # image, and the observation patch. No truncated rasterizer row list.
    g=scene.at(t)
    color=torch.ones(len(g.get_xyz),3,device='cuda',requires_grad=True)
    target=torch.ones(3,225,400,device='cuda');target[0]=torch.as_tensor(mask,device='cuda')
    yy,xx=torch.meshgrid(torch.arange(225,device='cuda'),torch.arange(400,device='cuda'),indexing='ij')
    target[2]=((xx-uv[0]).square()+(yy-uv[1]).square()<=4).float()
    package=renderer(cam,g,PIPE,scene.centers.new_zeros(3),0.,0.,0.,override_color=color)
    alpha=package['alpha'].squeeze().detach()
    assert (package['render'].detach()-alpha[None]).abs().max()<1e-6
    mass=torch.autograd.grad((package['render']*target).sum(),color)[0].detach()
    expected=(alpha[None]*target).sum((1,2))
    error=float(((mass.sum(0)-expected).abs()/expected.clamp_min(1)).max())
    assert error<5e-4 and mass.isfinite().all() and mass.min()>=-1e-7
    return mass,error


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-track-binding',action='store_true');a=p.parse_args()
    if not a.enable_track_binding:raise SystemExit('Track binding is opt-in')
    parent=RUNS/'S3/formal/checkpoint.pt';admission=read(REPORT/'S3/admission.json')
    assert admission['admitted'] and admission['native_images_reviewed'] and sha(parent)==admission['checkpoint_sha256']
    output=REPORT/'S4/bindings.json'
    if output.exists():
        assert read(output)['parent_sha256']==sha(parent)
        print('S4_BINDINGS_ALREADY_FROZEN',flush=True);return
    renderer=setup();torch.set_num_threads(4);scene,state,_=load_physics_state(parent)
    source=read(REPORT/'S4/image_tracks.json');meta=read(DATA/'transforms_train.json')
    frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    ledger=np.load(RUNS/'S2_observations.npz',allow_pickle=False)
    labels={int(n):(c,o) for n,c,o in zip(ledger['frame_ids'],ledger['components'],ledger['early_owner_to_component'])}
    accepted=[];rejected=[];maximum_error=0.
    for track in source['tracks']:
        k=track['local_early_object'];count=len(scene.objects[k].get_xyz)
        offset=len(scene.static.get_xyz)+sum(len(o.get_xyz) for o in scene.objects[:k])
        candidate=torch.ones(count,device='cuda',dtype=torch.bool);errors=[]
        for index,(n,uv) in enumerate(zip(track['frames'],track['uv_400'])):
            label,mapping=labels[n];component=int(mapping[k])
            if component<0:candidate[:]=False;break
            cam=camera(frames[n],meta,400);t=(n-1)/199
            with torch.no_grad():
                projected,valid=project(scene.object_at(k,t).get_xyz,cam)
                distance=(projected-projected.new_tensor(uv)).norm(dim=-1)
            candidate &= valid & (distance<=(1.5 if index==0 else 2.))
            if not bool(candidate.any()):break
            mass,error=visible_masses(renderer,scene,cam,t,label==component,uv);maximum_error=max(maximum_error,error)
            mass=mass[offset:offset+count]
            candidate &= eligible_rows(distance,valid,mass[:,0],mass[:,1],mass[:,2],index==0)
            errors.append(distance)
            if not bool(candidate.any()):break
        if not bool(candidate.any()):rejected.append({'track_id':track['track_id'],'reason':'No single row supports every qualified observation'});continue
        error=torch.stack(errors).mean(0);error[~candidate]=float('inf');row=int(error.argmin())
        item={**track,'local_row':row,'confidence':math.exp(-track['reverse_closure_px']),
            'binding_mean_reprojection_px':float(error[row]),'gaussian_row':offset+row,'binding_is_conditional':True}
        accepted.append(item)
        print('S4_ROW_CANDIDATE',track['track_id'],row,float(error[row]),flush=True)
    bad={x['track_id'] for x in accepted for y in accepted if x is not y and conflicts(x,y)}
    bindings=[x for x in accepted if x['track_id'] not in bad]
    rejected.extend({'track_id':n,'reason':'Conflicting observations on one immutable row'} for n in sorted(bad))
    write(output,{'parent_sha256':sha(parent),'parent_admission_sha256':sha(REPORT/'S3/admission.json'),
        'image_tracks_sha256':sha(REPORT/'S4/image_tracks.json'),'topology':[len(o.get_xyz) for o in scene.objects],
        'bindings':bindings,'rejected':rejected,'complete_mass_relative_error_max':maximum_error,
        'policy':'Fixed same-object row, source<=1.5px, all later<=2px, visible component mass>=.02, purity>=.95, patch mass>=.005; conflicts rejected',
        'heldout_used':False,'physical_material_ID_GT':False,'occlusion_reidentification':False})
    assert sha(parent)==admission['checkpoint_sha256']
    print('S4_BINDING_COMPLETE',len(bindings),'/',len(source['tracks']),flush=True)


if __name__=='__main__':main()
