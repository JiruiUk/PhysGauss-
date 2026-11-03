from types import SimpleNamespace
import numpy as np
import torch
from variants.staged_reconstruction.coarse_motion.core import make_controls,install
from variants.staged_reconstruction.sparse_motion.core import SparseDeform,matrix


class Field:
    reg_spatial_able=False
    reg_temporal_able=False

    def __init__(self): self.calls=[]; self.bias=torch.nn.Parameter(torch.tensor(.1))

    def step(self,x,t,fixed_attention=False):
        self.calls.append(len(x)); q=x.new_zeros(len(x),4); q[:,0]=1
        return {'d_xyz':torch.cat([q,torch.sin(x*3+t)*self.bias],-1),
                'd_rotation':q*0,'d_scaling':x*0,'reg_spatial':None,'reg_temporal':None}


def test_control_budget_and_uniform_bandwidth_are_deterministic():
    x=np.random.default_rng(4).normal(size=(200,3)).astype(np.float32)
    a,b=make_controls(x),make_controls(x)
    assert len(np.unique(a[0]))==16 and all(np.array_equal(i,j) for i,j in zip(a,b))
    assert np.all(a[2]==a[2][0]) and np.isclose(a[2][0],np.quantile(a[3],.9)*.5)


def test_disabled_install_does_not_read_assets_or_change_scene():
    field=Field(); scene=SimpleNamespace(deforms=[field])
    assert install(scene,{},False) is None and scene.deforms[0] is field
    assert install(scene,{'shared_motion_enabled':False},True) is None and scene.deforms[0] is field


def test_full_sparse_uses_only16_control_queries_and_has_gradients():
    x=np.random.default_rng(8).normal(size=(100,3)).astype(np.float32)
    _,c,r,_=make_controls(x)
    field=Field(); adapter=SparseDeform(field,torch.tensor(c),torch.tensor(r),mix=1.)
    query=torch.tensor(x,requires_grad=True)
    state=adapter.step(query,query.new_full((len(query),1),.3))
    assert field.calls==[16]
    position=(matrix(state['d_xyz'][:,:4])@query[...,None]).squeeze(-1)+state['d_xyz'][:,4:]
    position.square().mean().backward()
    assert query.grad.isfinite().all() and query.grad.abs().sum()>0
    assert field.bias.grad.isfinite() and field.bias.grad.abs()>0


def test_sparse_mapping_is_independent_of_query_order_and_chunking():
    x=np.random.default_rng(2).normal(size=(80,3)).astype(np.float32)
    _,c,r,_=make_controls(x); adapter=SparseDeform(Field(),torch.tensor(c),torch.tensor(r),mix=1.)
    q=torch.tensor(x)
    expected=adapter.step(q,q.new_full((len(q),1),.4))['d_xyz']
    chunked=torch.cat([adapter.step(a,a.new_full((len(a),1),.4))['d_xyz'] for a in q.split(17)])
    permuted=adapter.step(q.flip(0),q.new_full((len(q),1),.4))['d_xyz'].flip(0)
    assert torch.allclose(expected,chunked,atol=1e-6) and torch.allclose(expected,permuted,atol=1e-6)
