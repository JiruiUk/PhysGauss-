import numpy as np
import pytest
import torch
from variants.shape_coherence.model import fixed_graph,local_shape,clear_duck_masks


def fixture():
    gen=torch.Generator().manual_seed(17)
    x=torch.randn((60,3),generator=gen,dtype=torch.float64)*.03
    labels=torch.cat([torch.zeros(30,dtype=torch.long),torch.ones(30,dtype=torch.long)])
    a,n,v=fixed_graph(x,labels,anchors=16,k=12,radius=1.)
    return x,labels,a,n,v


def test_graph_does_not_cross_labels_or_connect_unknown():
    x,labels,*_=fixture();labels[:10]=-1
    a,n,v=fixed_graph(x,labels,16,12,1.)
    assert (labels[a]>=0).all() and (labels[n]==labels[a,None]).all()


def test_graph_is_deterministic_and_consumes_no_rng():
    x,labels,*_=fixture();state=torch.get_rng_state()
    first=fixed_graph(x,labels);second=fixed_graph(x,labels)
    assert all(torch.equal(a,b) for a,b in zip(first,second))
    assert torch.equal(state,torch.get_rng_state())


def test_full_affine_F_is_recovered():
    x,_,a,n,v=fixture();F=x.new_tensor([[1.2,.1,0],[0,.8,.1],[0,0,1.05]])
    r=local_shape(x,x@F.T+2,a,n,v)
    assert torch.allclose(r['F'],F.expand_as(r['F']),atol=1e-10)
    assert r['affine']<1e-20 and r['collapse']==0


def test_rigid_motion_is_unpenalized():
    x,_,a,n,v=fixture();R=x.new_tensor([[0,-1,0],[1,0,0],[0,0,1]])
    r=local_shape(x,x@R.T+1,a,n,v)
    assert r['affine']<1e-20 and r['collapse']==0


def test_collapse_is_not_hidden_by_zero_affine_residual():
    x,_,a,n,v=fixture();y=(x*.01).requires_grad_();r=local_shape(x,y,a,n,v)
    assert r['affine']<1e-20 and r['collapse']>0
    r['collapse'].backward();assert y.grad.isfinite().all() and y.grad.abs().sum()>0


def test_reflection_is_penalized():
    x,_,a,n,v=fixture();y=x*x.new_tensor([-1,1,1]);r=local_shape(x,y,a,n,v)
    assert r['collapse']>1


def test_reference_detached_but_nonaffine_positions_receive_gradients():
    x,_,a,n,v=fixture();ref=x.clone().requires_grad_();y=x.clone();y[0,0]+=.04;y.requires_grad_()
    r=local_shape(ref,y,a,n,v);(r['affine']+r['collapse']).backward()
    assert ref.grad is None and y.grad.isfinite().all() and y.grad.abs().sum()>0


def test_rank_deficient_surface_is_skipped():
    x,labels,*_=fixture();x[:,2]=0
    with pytest.raises(ValueError,match='degenerate'):fixed_graph(x,labels,16,12,1.)


def test_scene_masks_require_four_separated_candidates():
    rgba=np.zeros((80,80,4),dtype=np.uint8)
    for i in range(4):rgba[i*20+2:i*20+12,20:35]=[220,190,20,255]
    result=clear_duck_masks(rgba)
    assert set(np.unique(result))=={-1,0,1,2,3}
    with pytest.raises(ValueError):clear_duck_masks(rgba[:40])


def test_flags_off_returns_exact_parent_result_without_graph_access(tmp_path,monkeypatch):
    import json
    from variants.shape_coherence.runtime import Tuning,Parent
    config=tmp_path/'config.json';config.write_text(json.dumps({'enabled':True,'shape_enabled':False}))
    t=Tuning(config,tmp_path);sentinel=object()
    monkeypatch.setattr(Parent,'coverage_loss',lambda *a:sentinel)
    assert t.coverage_loss(20000,None,None,None,None) is sentinel
    assert t.graph is None and not hasattr(t,'shape_deform')


def test_disabled_new_module_inherits_original_image_and_physics_methods():
    from variants.shape_coherence.runtime import Tuning,Parent
    for method in ['image_loss','physics_time','time_step','regularization','kinematic_loss','after_update','evaluate']:
        assert getattr(Tuning,method) is getattr(Parent,method)


def test_graph_sidecar_is_bound_to_checkpoint_and_restores(tmp_path,monkeypatch):
    import json
    from variants.shape_coherence.runtime import Tuning,Parent
    from variants.metric_tuning.runtime import sha256
    config=tmp_path/'config.json';config.write_text(json.dumps({'enabled':True,'shape_enabled':True,'shape_start':10}))
    monkeypatch.setattr(Parent,'save',lambda *a:None)
    monkeypatch.setattr(Parent,'restore',lambda *a:None)
    p=tmp_path/'resume.pt';p.write_bytes(b'full checkpoint')
    p.with_suffix('.metrics.json').write_text('{}')
    tuning=Tuning(config,tmp_path);tuning.graph={'point_count':4,'frame_ids':[2,6,10]}
    tuning.save(p,{'iteration':20,'sha256':sha256(p)})
    restored=Tuning(config,tmp_path);restored.restore(p,20)
    assert restored.graph==tuning.graph
    with pytest.raises(ValueError,match='boundary'):restored.restore(p,21)
    p.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='boundary'):restored.restore(p,20)
    p.with_suffix('.shape.pt').write_bytes(b'corrupt graph')
    with pytest.raises(ValueError,match='SHA'):restored.restore(p,20)
