import torch
from variants.staged_reconstruction.stage4.bind import eligible_rows,conflicts


def test_invisible_or_impure_or_wrong_depth_rows_are_never_bound():
    error=torch.tensor([1.,1.,1.,1.,1.,1.])
    valid=torch.tensor([True,False,True,True,True,True])
    component=torch.tensor([.1,.1,.001,.09,.1,.1]);total=torch.full((6,),.1)
    patch=torch.tensor([.01,.01,.01,.01,0.,.01])
    result=eligible_rows(error,valid,component,total,patch,True)
    assert torch.equal(result,torch.tensor([True,False,False,False,False,True]))


def test_binding_conflict_requires_same_row_and_shared_observation():
    a={'local_early_object':0,'local_row':2,'frames':[2,3],'uv_400':[[1.,2.],[2.,2.]]}
    b={**a,'uv_400':[[1.,2.],[4.,2.]]}
    assert conflicts(a,b) and not conflicts(a,{**b,'local_row':3})
