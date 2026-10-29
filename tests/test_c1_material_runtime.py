"""Flags-off parity, training-time gradients and full material-state recovery."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from variants.c1_material.runtime import Tuning
from variants.color_repair.runtime import Tuning as Parent
from variants.metric_tuning.runtime import sha256


class Fields(torch.nn.Module):
    def __init__(self):
        super().__init__();self.head=torch.nn.Linear(4,9,dtype=torch.float64)
    def forward(self,coords,indices=None):
        value=self.head(coords);return value[:,:3],value[:,3:]


def fixture(tmp_path,learn=True):
    cfg={'enabled':True,'material_enabled':True,'learn_young':learn,'ack_global_effective_material':True,
         'material_start':0,'material_ramp':1,'material_interval':1,'material_samples':32}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    tuning=Tuning(path,tmp_path)
    axis=torch.linspace(-.3,.3,4,dtype=torch.float64);X=torch.cartesian_prod(axis,axis,axis)
    gaussian=SimpleNamespace(get_xyz=X,particle_indices=torch.arange(len(X)),velocity_net=Fields(),
                             force_sigma=torch.tensor(.1,dtype=torch.float64,requires_grad=True))
    tuning.after_scene(gaussian,None);tuning.train_names={'0002'}
    return tuning,gaussian,path


def test_material_off_is_exact_parent_loss_without_new_rng_or_state(tmp_path):
    p=tmp_path/'off.json';p.write_text(json.dumps({'enabled':True,'mask_weight':.03,'foreground_rgb_weight':.1,
                                                'mask_start':20,'mask_ramp':10,'material_enabled':False}))
    before=torch.get_rng_state().clone();a=Tuning(p,tmp_path);b=Parent(p,tmp_path)
    assert torch.equal(before,torch.get_rng_state()) and a.material is None and a.material_optimizer is None
    a.train_names=b.train_names={'0002'}
    view=SimpleNamespace(image_name='0002',coverage_target=torch.ones(1,2,3),original_image=torch.ones(3,2,3))
    pkg={'alpha':torch.zeros(1,2,3,requires_grad=True),'render':torch.zeros(3,2,3,requires_grad=True)}
    assert torch.equal(a.coverage_loss(25,view,pkg,None,None),b.coverage_loss(25,view,pkg,None,None))


def test_real_runtime_loss_reaches_material_and_trajectory(tmp_path):
    tuning,g,_=fixture(tmp_path)
    A=torch.diag(torch.tensor([1.1,.95,1.02],dtype=torch.float64)).requires_grad_()
    translation=g.get_xyz@A.T-g.get_xyz
    q=torch.zeros(len(translation),4,dtype=torch.float64);q[:,0]=1
    warp=torch.cat((q,translation),-1)
    view=SimpleNamespace(image_name='0002',fid=torch.tensor([.4],dtype=torch.float64))
    loss=tuning.coverage_loss(1,view,{},g,warp);loss.backward()
    assert A.grad.abs().sum()>0 and tuning.material.raw_E.grad.abs()>0
    tuning.after_update(1,g)
    assert tuning.material_updates==1 and abs(tuning.material.young.item()-1)>1e-7


@pytest.mark.parametrize('learn',[False,True])
def test_checkpoint_restores_E_optimizer_and_graph_and_rejects_tamper(tmp_path,learn):
    a,g,path=fixture(tmp_path,learn)
    if learn:
        (a.material.young-2).square().backward();a.after_update(1,g)
    a.material_graph.get(g.get_xyz,1)
    checkpoint=tmp_path/'resume.pt';checkpoint.write_bytes(b'isolated full state fixture')
    a.save(checkpoint,{'sha256':sha256(checkpoint),'iteration':1})
    b=Tuning(path,tmp_path);b.after_scene(g,None);b.train_names={'0002'};b.restore(checkpoint,1)
    assert torch.equal(a.material.young,b.material.young)
    assert b.material_updates==a.material_updates
    assert torch.equal(a.material_graph.neighbors,b.material_graph.neighbors)
    if learn:
        pa=next(iter(a.material_optimizer.state.values()));pb=next(iter(b.material_optimizer.state.values()))
        assert torch.equal(pa['exp_avg'],pb['exp_avg']) and torch.equal(pa['step'],pb['step'])
    sidecar=checkpoint.with_suffix('.material.pt');sidecar.write_bytes(sidecar.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):b.restore(checkpoint,1)


def test_scope_must_acknowledge_missing_true_material_labels(tmp_path):
    p=tmp_path/'config.json';p.write_text(json.dumps({'enabled':True,'material_enabled':True}))
    with pytest.raises(ValueError,match='effective prior'):Tuning(p,tmp_path)


def test_fixed_and_learned_configs_differ_only_in_parameter_learning():
    from variants.c1_material.run import configuration,NAMES
    a,ta=configuration(NAMES[1]);b,tb=configuration(NAMES[2])
    assert ta==tb and {k for k in a if a[k]!=b[k]}=={'learn_young'}
    assert a['material_enabled'] and a['isotropic_weight']==a['kinematic_weight']==0


def test_fork_only_changes_optional_runtime_import():
    from variants.c1_material.prepare import prepare
    from variants.color_repair.prepare import prepare as parent
    a,b=prepare(),parent()
    assert (a/'train_pidg.py').read_text().replace('from variants.c1_material.runtime import Tuning',
        'from variants.color_repair.runtime import Tuning')==(b/'train_pidg.py').read_text()
    assert (a/'gaussian_renderer/__init__.py').read_bytes()==(b/'gaussian_renderer/__init__.py').read_bytes()
