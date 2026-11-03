"""Engineering probe only; experimental S3 training still requires S2 admission."""
import argparse
import random
import numpy as np
import torch
from ..prepare_data import ROOT,REPORT,RUNS,DATA,read,write,sha
from ..train_static import setup,camera
from ..stage2.train import image,objective
from ..recovery.train import RecoveryScene
from ..recovery.prepare import REC_REPORT,REC_RUNS
from .physics import PhysicsField


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-physics-probe',action='store_true');args=p.parse_args()
    if not args.enable_physics_probe:raise SystemExit('S3 probe is opt-in; no baseline import/render')
    torch.set_num_threads(4);torch.manual_seed(20261105);random.seed(20261105);np.random.seed(20261105)
    renderer=setup();scene=RecoveryScene(read(REC_REPORT/'S2_protocol.json'))
    source=REC_RUNS/'_S2_smoke/checkpoint.pt';source_hash=sha(source)
    checkpoint=torch.load(source,map_location='cpu',weights_only=False);scene.load_state_dict(checkpoint['model'])
    field=PhysicsField(4);opt=torch.optim.Adam(field.parameters(),lr=.001)
    frozen={k:v.clone() for k,v in scene.static.state_dict().items()}
    frames=read(DATA/'transforms_train.json');frame=frames['frames'][0];rgba=image(frame,800)
    target=torch.tensor(rgba[...,:3]*rgba[...,3:4],device='cuda').permute(2,0,1)
    rgb=objective(scene.render(renderer,camera(frame,frames,800),frame['time'])['render'],target)
    terms,rows=field.loss(scene,frame['time'],1,samples=16)
    physical=.01*terms['kinematic']+.00001*terms['CMR']
    modules={'fields':list(field.parameters()),'deformation':list(scene.networks.parameters()),'grid':list(scene.grids.parameters()),
             'canonical':[o._xyz for o in scene.objects]}
    gradients={}
    for name,params in modules.items():
        g=torch.autograd.grad(physical,params,allow_unused=True,retain_graph=True)
        assert all(v is None or v.isfinite().all() for v in g)
        gradients[name]=sum(float(v.detach().abs().sum()) for v in g if v is not None)
        assert gradients[name]>0,(name,gradients)
    before=[p.detach().clone() for p in field.parameters()]
    physical.backward();opt.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,field.parameters()))
    assert all(p.grad is None for p in scene.static.parameters())
    assert all(torch.equal(v,scene.static.state_dict()[k]) for k,v in frozen.items())
    assert sha(source)==source_hash
    directory=REPORT/'S3';directory.mkdir(parents=True,exist_ok=True)
    write(directory/'engineering_probe.json',{'passed':True,'not_formal_training_or_S2_admission':True,
        'source_smoke_checkpoint_sha256':source_hash,'source_smoke_step':checkpoint['iteration'],
        'physics_gradient_L1':gradients,'raw_terms':{k:float(v.detach()) for k,v in terms.items()},
        'static_tensors_unchanged':True,'static_physics_samples':0,'cross_object_neighbors':0,
        'no_body_force_contact_or_constitutive_loss':True,'independent_velocity_and_predicted_stress_preserved':True,
        'actual_time':'(original_frame-1)/199, gaps determined by metadata; physical acquisition seconds unavailable',
        'diagnostics':rows})
    print('S3_ENGINEERING_PROBE_PASSED',gradients,flush=True)


if __name__=='__main__':main()
