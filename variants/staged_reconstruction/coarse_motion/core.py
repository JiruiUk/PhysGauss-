"""Small shared continuous control field; control count is not body count."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from ..prepare_data import ROOT,sha
from ..sparse_motion.core import controls_from_canonical,SparseDeform


def make_controls(points,count=16):
    rows,nodes,_=controls_from_canonical(points,count)
    nearest=cKDTree(nodes).query(np.asarray(points),k=1,workers=1)[0]
    # Uniform bandwidth avoids very large radii around isolated FPS controls.
    # It is fixed from initial coverage, not tuned on model/heldout performance.
    radius=max(float(np.quantile(nearest,.9))*.5,1e-4)
    return rows,nodes,np.full(count,radius,np.float32),nearest


def install(scene,plan,enabled=False):
    if not enabled or not plan.get('shared_motion_enabled',False): return None
    path=ROOT/plan['controls']; assert sha(path)==plan['controls_sha256']
    with np.load(path) as data:
        adapter=SparseDeform(scene.deforms[0],torch.tensor(data['xyz'],device=scene.centers.device),
                             torch.tensor(data['radius'],device=scene.centers.device),mix=1.)
    scene.deforms[0]=adapter
    return adapter
