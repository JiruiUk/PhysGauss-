"""Verify original CUDA interpolation against complete Torch higher derivatives."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repo=args.repo.resolve()
    sys.path.insert(0,str(repo));os.chdir(repo)
    import torch
    from hashencoder.hashgrid import HashEncoder
    from physics.hash_autograd import higher_order_hash
    from physics.kinematics import compute_velocity_acceleration,official_position,compute_velocity_from_deformation
    from scene_PIDG.deform_model import DeformModel
    torch.manual_seed(43)
    encoder=HashEncoder(input_dim=3,num_levels=3,level_dim=2,base_resolution=4,
                        desired_resolution=8,log2_hashmap_size=8).cuda()
    points=torch.tensor([[.2357,.3871,-.1234],[.12,-.35,.54]],device='cuda',requires_grad=True)
    cuda=encoder(points)
    with higher_order_hash(): pure=encoder(points)
    torch.testing.assert_close(pure,cuda,atol=1e-7,rtol=2e-4)
    dcuda=torch.autograd.grad(cuda.sum(),points,create_graph=True)[0]
    dpure=torch.autograd.grad(pure.sum(),points,create_graph=True)[0]
    torch.testing.assert_close(dpure,dcuda,atol=1e-6,rtol=2e-3)
    second=torch.autograd.grad(dpure.sum(),points,create_graph=True)[0]
    second.square().mean().backward()
    assert torch.isfinite(encoder.embeddings.grad).all() and encoder.embeddings.grad.abs().sum()>0
    grid={'canonical_num_levels':3,'canonical_level_dim':2,'canonical_base_resolution':4,
          'canonical_desired_resolution':8,'canonical_log2_hashmap_size':8,
          'deform_num_levels':3,'deform_level_dim':2,'deform_base_resolution':[4,4,4],
          'deform_desired_resolution':[8,8,8],'deform_log2_hashmap_size':8,'bound':1.6}
    deform=DeformModel(grid,{'depth':0,'width':16,'directional':True},
                       reg_spatial_able=False,reg_temporal_able=False)
    X=torch.rand(8,3,device='cuda')*.5
    t=torch.tensor([.2357],device='cuda',requires_grad=True)
    x,v,a=compute_velocity_acceleration(lambda X,t:official_position(deform,X,t),X,t)
    assert x.shape==v.shape==a.shape==(8,3)
    a.square().mean().backward()
    for p in list(deform.grid4d.parameters())+list(deform.deform.parameters()):
        if p.grad is not None: assert torch.isfinite(p.grad).all()
    # Stage 2B/flow uses CUDA first derivatives on full particle batches. Its
    # parameter backward must match the complete Torch path before opening the gate.
    parameters=list(deform.grid4d.parameters())+list(deform.deform.parameters())
    def velocity_gradients(higher):
        velocity=compute_velocity_from_deformation(lambda X,t:official_position(deform,X,t,higher),X,t)
        gradients=torch.autograd.grad(velocity.square().sum(),parameters,allow_unused=True)
        return velocity,gradients
    fast_velocity,fast_gradients=velocity_gradients(False)
    exact_velocity,exact_gradients=velocity_gradients(True)
    torch.testing.assert_close(fast_velocity,exact_velocity,atol=1e-6,rtol=1e-3)
    gradient_error=0.
    for fast,exact in zip(fast_gradients,exact_gradients):
        assert (fast is None)==(exact is None)
        if exact is not None:
            assert torch.isfinite(fast).all() and torch.isfinite(exact).all()
            torch.testing.assert_close(fast,exact,atol=1e-7,rtol=3e-3)
            gradient_error=max(gradient_error,float((fast-exact).detach().abs().max()))
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from pidg_benchmark import write_json
    write_json(args.output,{'cuda_torch_forward_max_error':float((pure-cuda).detach().abs().max()),
              'cuda_torch_first_derivative_max_error':float((dpure-dcuda).detach().abs().max()),
              'full_hash_second_coordinate_and_parameter_derivatives':True,
              'full_quaternion_trajectory_acceleration_backward':True,
              'cuda_flow_velocity_parameter_gradient_max_error':gradient_error,'arbitrary_time':.2357})
    print('PASS: CUDA/Torch values and first derivatives; full trajectory acceleration backward')


if __name__=='__main__': main()
