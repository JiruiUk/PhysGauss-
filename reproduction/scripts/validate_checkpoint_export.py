from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch

root = Path('F:/A_Edinburgh/PhysGauss')
os.chdir(root / 'third_party/PIDG')
sys.path.insert(0, str(root / 'reproduction/scripts'))
sys.path.insert(0, str(root / 'third_party/PIDG'))
from training_checkpoint import load_state, restore_training_state, cpu_tree
from scene_PIDG import GaussianModel, DeformModel

checkpoint = root / 'runs/pidg_official/motion_kuro/checkpoints/resume.pt'
manifest = json.loads(checkpoint.with_suffix('.json').read_text())
with checkpoint.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['sha256']
state = load_state(checkpoint)
dataset, opt = Namespace(**state['dataset']), Namespace(**state['opt'])
g = GaussianModel(dataset.sh_degree)
d = DeformModel(dataset.grid_args, dataset.network_args,
    spatial_downsample_ratio=opt.spatial_downsample_ratio, spatial_perturb_range=opt.spatial_perturb_range,
    temporal_downsample_ratio=opt.temporal_downsample_ratio, temporal_perturb_range=opt.temporal_perturb_range,
    scale_xyz=dataset.scale_xyz, reg_spatial_able=opt.lambda_spatial_tv > 0,
    reg_temporal_able=opt.lambda_temporal_tv > 0)
d.train_setting(opt)
class Scene:
    def __init__(self):
        self.train = [Namespace(image_name=n) for n in reversed(state['train_camera_names'])]
        self.test = [Namespace(image_name=n) for n in reversed(state['test_camera_names'])]
    def getTrainCameras(self): return self.train
    def getTestCameras(self): return self.test
scene = Scene()
restored = restore_training_state(checkpoint, g, d, opt, scene)

def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for av, bv in zip(a, b): equal(av, bv)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    else:
        assert a == b, (a, b)

for name, tensor in state['gaussian_parameters'].items(): equal(getattr(g, name), tensor)
for name, tensor in state['gaussian_buffers'].items(): equal(getattr(g, name), tensor)
equal(g.optimizer.state_dict(), state['gaussian_optimizer'])
equal(d.optimizer.state_dict(), state['deform_optimizer'])
equal(g.velocity_net.state_dict(), state['velocity_net'])
equal(d.grid4d.state_dict(), state['grid4d'])
equal(d.deform.state_dict(), state['deform'])
equal(random.getstate(), state['rng_python'])
equal(np.random.get_state(), state['rng_numpy'])
equal(torch.get_rng_state(), state['rng_torch'])
equal(torch.cuda.get_rng_state_all(), state['rng_cuda'])
assert [c.image_name for c in scene.train] == state['train_camera_names']
assert [scene.train.index(c) for c in restored['viewpoint_stack']] == state['viewpoint_stack']
assert len(g.optimizer.state) > 0 and len(d.optimizer.state) > 0
with torch.no_grad():
    sample = g.get_xyz[:16].detach()
    prediction = d.step(sample, torch.full((16, 1), 0.5, device='cuda'))
    assert all(torch.isfinite(prediction[k]).all() for k in ('d_xyz','d_rotation','d_scaling'))
    v, stress = g.get_velocity_and_stress(torch.tensor([0.5],device='cuda'))
    assert torch.isfinite(v).all() and torch.isfinite(stress).all()
torch.cuda.synchronize()
report = {'status':'passed','iteration':state['iteration'],'sha256':manifest['sha256'],
    'gaussian_parameters':len(state['gaussian_parameters']),
    'gaussian_optimizer_entries':len(g.optimizer.state), 'deform_optimizer_entries':len(d.optimizer.state),
    'optimizer_states_exact':True,'random_states_exact':True,'camera_stack_restored':True,
    'forward_finite':True,'validated_in_new_process':True,'torch_version':str(torch.__version__)}
checkpoint.with_name('validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
