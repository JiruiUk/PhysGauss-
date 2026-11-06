import torch
import json
from physics.kinematics import apply_trajectory_constraints,quaternion_matrix
from physics.deformation_gradient import compute_deformation_gradient


def test_translation_uniform_scaling_and_shear():
    X=torch.randn(6,3,dtype=torch.float64)
    I=torch.eye(3,dtype=X.dtype).expand(6,3,3)
    F,J=compute_deformation_gradient(lambda X,t:X+t,X,.2357)
    torch.testing.assert_close(F,I)
    torch.testing.assert_close(J,torch.ones(6,1,dtype=X.dtype))
    F,J=compute_deformation_gradient(lambda X,t:X*(1+t),X,.2)
    torch.testing.assert_close(F,I*1.2)
    torch.testing.assert_close(J,torch.full((6,1),1.2**3,dtype=X.dtype))
    shear=torch.eye(3,dtype=X.dtype); shear[0,1]=.2
    F,J=compute_deformation_gradient(lambda X,t:X@shear.T,X,.2)
    torch.testing.assert_close(F,shear.expand_as(F))
    torch.testing.assert_close(J,torch.ones_like(J))


def test_canonical_anchor_has_identity_gradient(tmp_path,monkeypatch):
    path=tmp_path/'config.json'
    path.write_text(json.dumps({'anchor_deformation':True}))
    monkeypatch.setenv('PIDG_UPGRADE_CONFIG',str(path))
    X=torch.randn(5,3,dtype=torch.float64)
    def position(X,t):
        q=torch.cat([torch.ones_like(t),.01*X],-1)
        raw=torch.cat([q,torch.sin(X)],-1)
        motion=apply_trajectory_constraints(raw,t)
        return (quaternion_matrix(motion[:,:4])@X[...,None]).squeeze(-1)+motion[:,4:]
    F,J=compute_deformation_gradient(position,X,0.)
    torch.testing.assert_close(F,torch.eye(3,dtype=X.dtype).expand(5,3,3))
    torch.testing.assert_close(J,torch.ones(5,1,dtype=X.dtype))
