import torch
from physics.kinematics import compute_velocity_from_deformation, quaternion_matrix


def test_translation_and_full_rotational_velocity():
    X = torch.randn(5,3,dtype=torch.float64)
    v = torch.tensor([.2,-.3,.4],dtype=X.dtype)
    result = compute_velocity_from_deformation(lambda X,t:X+v*t,X,.2357)
    torch.testing.assert_close(result,v.expand_as(X))
    def rotate(X,t):
        q = torch.cat([torch.cos(t/2),t*0,t*0,torch.sin(t/2)],-1)
        return (quaternion_matrix(q)@X[...,None]).squeeze(-1)
    t=.2357
    x = X[:,0]*torch.cos(torch.tensor(t))+ -X[:,1]*torch.sin(torch.tensor(t))
    y = X[:,0]*torch.sin(torch.tensor(t))+X[:,1]*torch.cos(torch.tensor(t))
    expected = torch.stack([-y,x,X[:,2]*0],-1)
    torch.testing.assert_close(compute_velocity_from_deformation(rotate,X,t),expected,atol=1e-7,rtol=1e-7)
