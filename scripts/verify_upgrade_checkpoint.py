"""Small real-CUDA restart test, including all new physics optimizer parameters."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=args.repo.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(repo));os.chdir(repo)
    import numpy as np
    import torch
    from arguments import OptimizationParams
    from argparse import ArgumentParser
    from scene_PIDG import GaussianModel,DeformModel
    from utils.graphics_utils import BasicPointCloud
    from physics.kinematics import official_position
    from training_checkpoint import save_training_state,restore_training_state,cpu_tree
    p=ArgumentParser();op=OptimizationParams(p);opt=op.extract(p.parse_args([]))
    grid={'canonical_num_levels':3,'canonical_level_dim':2,'canonical_base_resolution':4,
          'canonical_desired_resolution':8,'canonical_log2_hashmap_size':8,
          'deform_num_levels':3,'deform_level_dim':2,'deform_base_resolution':[4,4,4],
          'deform_desired_resolution':[8,8,8],'deform_log2_hashmap_size':8,'bound':1.6}
    rng=np.random.default_rng(7)
    points=rng.uniform(-.2,.2,(24,3)).astype(np.float32)
    cloud=BasicPointCloud(points=points,colors=np.full_like(points,.6),normals=np.zeros_like(points))
    def models():
        g=GaussianModel(3);g.create_from_pcd(cloud,1.);g.training_setup(opt)
        d=DeformModel(grid,{'depth':0,'width':16,'directional':True},reg_spatial_able=False,reg_temporal_able=False)
        d.train_setting(opt)
        cameras=[SimpleNamespace(image_name=f'camera{i}') for i in range(3)]
        scene=SimpleNamespace(getTrainCameras=lambda:cameras,getTestCameras=lambda:[])
        return g,d,scene
    def step(g,d):
        g.optimizer.zero_grad(set_to_none=True);d.optimizer.zero_grad(set_to_none=True)
        t=g.get_xyz.new_tensor(.2357)
        X=g.get_xyz
        position=official_position(d,X,t.expand(len(X),1))
        target=X.detach()+X.new_tensor([.01,.02,-.01])
        physics,_=g.physics(g,d,t,100,output)
        loss=(position-target).square().mean()+physics
        if hasattr(g.physics,'flow_uncertainty'):
            loss=loss+.01*g.physics.flow_uncertainty(position[:,:2]).mean()
        loss.backward()
        g.optimizer.step();d.optimizer.step()
        g.update_learning_rate(100);d.update_learning_rate(100)
        return float(loss.detach())
    torch.manual_seed(71)
    g,d,scene=models()
    step(g,d)
    local={'gaussians':g,'deform':d,'scene':scene,'iteration':100,'dataset':SimpleNamespace(),
           'opt':opt,'pipe':SimpleNamespace(),'viewpoint_stack':scene.getTrainCameras()[1:],
           'ema_loss_for_log':.1,'best_psnr':1.,'best_iteration':100,
           'testing_iterations':[120],'saving_iterations':[120]}
    save_training_state(local,output/'resume.pt')
    reference_loss=step(g,d)
    reference=cpu_tree({'physics':g.physics.state_dict(),'grid':d.grid4d.state_dict(),
                       'deform':d.deform.state_dict(),'velocity':g.velocity_net.state_dict(),
                       'optimizer_g':g.optimizer.state_dict(),'optimizer_d':d.optimizer.state_dict(),
                       'xyz':g.get_xyz})
    g2,d2,scene2=models()
    restored=restore_training_state(output/'resume.pt',g2,d2,opt,scene2)
    assert restored['iteration']==100 and len(restored['viewpoint_stack'])==2
    resumed_loss=step(g2,d2)
    actual=cpu_tree({'physics':g2.physics.state_dict(),'grid':d2.grid4d.state_dict(),
                    'deform':d2.deform.state_dict(),'velocity':g2.velocity_net.state_dict(),
                    'optimizer_g':g2.optimizer.state_dict(),'optimizer_d':d2.optimizer.state_dict(),
                    'xyz':g2.get_xyz})
    largest=0.
    def compare(a,b):
        nonlocal largest
        if torch.is_tensor(a):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            if a.is_floating_point(): largest=max(largest,float((a-b).abs().max()))
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a: compare(a[key],b[key])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b): compare(x,y)
        else: assert a==b
    compare(reference,actual)
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from pidg_benchmark import write_json
    write_json(output/'verification.json',{'passed':True,'reference_loss':reference_loss,
               'resumed_loss':resumed_loss,'largest_state_difference':largest,
               'checked':['Gaussians','velocity','material/MoE/residual/uncertainty','spline',
                          'hash embeddings','both Adam optimizers','RNG sampling','camera stack']})
    print('PASS: new physics and spline state, both optimizers, RNG and camera order resume')


if __name__=='__main__': main()
