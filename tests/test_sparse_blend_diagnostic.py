import torch
from scripts.audit_sparse_blend import differential
from variants.staged_reconstruction.sparse_motion.core import weights,interpolate,matrix


def data():
    c=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=torch.double)
    radius=torch.ones(4,dtype=torch.double)*.5
    q=torch.tensor([[1.,.1,.2,.3]],dtype=torch.double).expand(4,4).clone()
    node={'d_xyz':torch.cat([q,torch.tensor([[.1,.2,.3]],dtype=torch.double).expand(4,3)],-1),
          'd_rotation':torch.zeros(4,4,dtype=torch.double),'d_scaling':torch.zeros(4,3,dtype=torch.double)}
    return c,radius,node


def test_constant_rigid_transform_derivative_is_rotation():
    c,r,node=data(); x=torch.tensor([[.2,.3,.1]],dtype=torch.double)
    blend,full=differential(x,c,r,node)
    assert torch.allclose(blend,full,atol=1e-12)
    assert torch.allclose(full,matrix(node['d_xyz'][0,:4])[None],atol=1e-12)


def test_full_weight_derivative_matches_autograd_for_varying_transforms():
    c,r,node=data(); node['d_xyz'][1,4]+=.3; node['d_xyz'][2,2]+=.2
    x=torch.tensor([[.2,.3,.1]],dtype=torch.double,requires_grad=True)
    _,analytic=differential(x,c,r,node)
    def actual(query):
        ids,w=weights(query,c,r); state=interpolate(query,node,ids,w)
        return (matrix(state['d_xyz'][:,:4])@query[...,None]).squeeze(-1)+state['d_xyz'][:,4:]
    jacobian=torch.autograd.functional.jacobian(actual,x)[0,:,0,:]
    assert torch.allclose(analytic[0],jacobian,atol=1e-10)
