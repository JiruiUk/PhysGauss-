import pytest
import torch
from variants.staged_reconstruction.i10_joint.core import merge_model


def values():
    a={'static._opacity':torch.tensor([1.]),'pools.0._xyz':torch.tensor([2.]),'physics.value':torch.tensor([3.]),'centers':torch.zeros(1,3)}
    b={key:value+4 if key!='centers' else value.clone() for key,value in a.items()}
    return a,b


def test_off_original_object_identity():
    a,b=values();assert merge_model(a,b) is a


def test_static_only_source_replacement_and_no_mutation():
    a,b=values();result=merge_model(a,b,True)
    assert result is not a and result['static._opacity'] is b['static._opacity']
    assert all(result[key] is a[key] for key in a if not key.startswith('static.'))
    assert a['static._opacity'].item()==1 and b['static._opacity'].item()==5


def test_physics_and_canonical_source_kept():
    a,b=values();result=merge_model(a,b,True)
    assert result['physics.value'].item()==3 and result['pools.0._xyz'].item()==2


def test_coordinate_frame_mismatch_rejected():
    a,b=values();b['centers']=torch.ones(1,3)
    with pytest.raises(AssertionError,match='coordinate frame'):merge_model(a,b,True)


def test_schema_mismatch_rejected():
    a,b=values();b['extra']=torch.ones(1)
    with pytest.raises(AssertionError,match='schema'):merge_model(a,b,True)


@pytest.mark.parametrize('other',[torch.ones(2),torch.ones(1,dtype=torch.float64)])
def test_shape_and_dtype_mismatch_rejected(other):
    a,b=values();b['static._opacity']=other
    with pytest.raises(AssertionError,match='static._opacity'):merge_model(a,b,True)
