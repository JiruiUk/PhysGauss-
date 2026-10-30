"""Color is opt-in and may separate otherwise identical motion groups."""
import numpy as np
import pytest

from variants.object_centers.adaptive_grouping import density_motion,spatial_motion_graph
from variants.object_centers.color_features import motion_color_features


def inputs():
    rng=np.random.default_rng(7);n=100;times=np.array([0.,.2,.7,1.])
    x=rng.normal(0,.001,(n,3));poses=x[None]+times[:,None,None]*np.array([.1,0,0])[None,None]
    colors=np.r_[np.tile([.8,.7,.1],(50,1)),np.tile([.1,.3,.8],(50,1))]
    colors+=rng.normal(0,.005,colors.shape)
    return poses,times,colors,np.ones(n,dtype=bool),np.ones(n)


def test_color_features_append_weighted_rgb_and_zero_is_exact_motion_path():
    from variants.object_centers.motion_grouping import motion_features
    poses,times,colors,_,_=inputs()
    expected=motion_features(poses,times,'displacement')
    assert np.array_equal(motion_color_features(poses,times,'displacement',None,0),expected)
    actual=motion_color_features(poses,times,'displacement',colors,.75)
    assert np.array_equal(actual[:,:expected.shape[1]],expected)
    assert np.allclose(actual[:,-3:],colors*.75)


def test_color_density_separates_same_motion_different_colors_without_target_count():
    poses,times,colors,eligible,weights=inputs()
    labels,stats,_=density_motion(poses,times,eligible,weights,'displacement',
        min_cluster_size=15,min_samples=5,colors=colors,color_weight=1.)
    assert stats['actual_groups']==2
    assert len(set(labels[:50]))==len(set(labels[50:]))==1 and labels[0]!=labels[50]
    assert stats['requested_group_count'] is None


def test_color_disabled_density_reproduces_labels_and_membership_exactly():
    poses,times,colors,eligible,weights=inputs()
    a=density_motion(poses,times,eligible,weights,'displacement',min_cluster_size=15,min_samples=5)
    b=density_motion(poses,times,eligible,weights,'displacement',min_cluster_size=15,min_samples=5,
                     colors=colors,color_weight=0.)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[2],b[2]) and a[1]==b[1]


def test_color_graph_separates_spatially_mixed_same_motion_colors():
    poses,times,colors,eligible,weights=inputs()
    labels,stats=spatial_motion_graph(poses,times,eligible,weights,min_cluster_size=10,
        neighbors=99,radius_factor=100,radius_cap=1.,colors=colors,color_radius=.2)
    assert stats['actual_groups']==2 and labels[0]!=labels[50]
    assert stats['color_compatible_edges']<stats['motion_compatible_edges']


def test_color_graph_disabled_is_exact_legacy_path():
    poses,times,colors,eligible,weights=inputs()
    a=spatial_motion_graph(poses,times,eligible,weights,min_cluster_size=10)
    b=spatial_motion_graph(poses,times,eligible,weights,min_cluster_size=10,colors=colors,color_radius=None)
    assert np.array_equal(a[0],b[0]) and a[1]==b[1]


def test_invalid_active_color_features_are_rejected():
    poses,times,colors,_,_=inputs();colors[0,0]=np.nan
    with pytest.raises(ValueError):motion_color_features(poses,times,'displacement',colors,1.)
    with pytest.raises(ValueError):motion_color_features(poses,times,'displacement',np.zeros((1,3)),1.)
