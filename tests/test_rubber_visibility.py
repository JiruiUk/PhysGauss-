import numpy as np
from variants.staged_reconstruction.rubber_visibility.core import select_ids,weighted_observation,metadata_census,packing_colors

def test_invalid_indices_and_first20_not_all():
    ids=np.array([[[-1,2],[4,1]],[[0,9],[2,-1]]]);mask=np.array([[True,False],[True,False]])
    assert np.array_equal(select_ids(ids,mask,5),[0,2,4])
    assert select_ids(ids,np.zeros((2,2),bool),5).size==0

def test_zero_visible_weight_unknown():
    row=weighted_observation(np.zeros((2,2)),np.ones((2,2,3)))
    assert row['visible_weight']==0 and row['observed_mean_RGB'] is None and row['weight_on_observed_RGB_lt005'] is None

def test_weighted_source_rgb_is_descriptive():
    weight=np.array([[1.,3.]]);rgb=np.array([[[0.,0.,0.],[1.,1.,1.]]])
    row=weighted_observation(weight,rgb)
    assert row=={'visible_weight':4.,'observed_mean_RGB':.75,'weight_on_observed_RGB_lt005':.25}

def test_groups_overlap_without_partition_assumption():
    colors=packing_colors(5,2,[np.array([0,1]),np.array([1,4])])
    assert np.array_equal(colors[:2],np.zeros((2,3)))
    assert np.array_equal(colors[3],[1.,1.,0.]) and np.array_equal(colors[6],[0.,1.,0.])

def test_exact_metadata_times_not_nearest_time_equivalence():
    identity=np.eye(4).tolist();other=np.eye(4);other[0,3]=1
    frames=[{'time':0.,'transform_matrix':identity},{'time':0.,'transform_matrix':other.tolist()},
        {'time':1e-12,'transform_matrix':identity}]
    row=metadata_census(frames)
    assert row['frames']==3 and row['unique_times']==2 and row['same_time_multicamera_times']==1
