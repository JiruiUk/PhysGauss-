"""No target cluster count: homogeneous, two-body and three-body cases."""
import inspect
import numpy as np
import pytest

from variants.object_centers.adaptive_grouping import density_motion,spatial_motion_graph


def synthetic(groups):
    rng=np.random.default_rng(11);n=50
    times=np.array([0.,.2,.65,1.])
    initial=rng.normal(0,.01,(groups*n,3))
    directions=np.array([[1,0,0],[-1,0,0],[0,1,0]])[:groups]
    velocity=np.repeat(directions,n,axis=0)+rng.normal(0,.01,(groups*n,3))
    return initial[None]+times[:,None,None]*velocity[None],times


@pytest.mark.parametrize('count',[2,3])
def test_density_discovers_different_counts_with_same_parameters(count):
    poses,times=synthetic(count);n=poses.shape[1]
    labels,stats,confidence=density_motion(poses,times,np.ones(n,dtype=bool),np.ones(n),
        'displacement',min_cluster_size=10,min_samples=5)
    assert stats['actual_groups']==count
    for g in range(count):assert len(set(labels[g*50:(g+1)*50]))==1
    assert np.isfinite(confidence).all() and ((confidence>=0)&(confidence<=1)).all()


def test_density_allows_one_static_group_and_keeps_ineligible_unknown():
    poses=np.zeros((3,40,3));eligible=np.ones(40,dtype=bool);eligible[-1]=False
    labels,stats,confidence=density_motion(poses,np.array([0.,.4,1.]),eligible,np.ones(40),
        'displacement',min_cluster_size=10,min_samples=5)
    assert stats['actual_groups']==1 and labels[-1]==-1 and confidence[-1]==0


def test_density_too_few_points_returns_unknown_instead_of_forced_cluster():
    labels,stats,_=density_motion(np.zeros((2,4,3)),np.array([0.,1.]),np.ones(4,dtype=bool),
        np.ones(4),'displacement',min_cluster_size=8,min_samples=5)
    assert (labels==-1).all() and stats['actual_groups']==0


def test_direct_graph_splits_separated_static_bodies_without_base_labels():
    rng=np.random.default_rng(3);x=np.r_[rng.normal(0,.005,(20,3)),rng.normal(0,.005,(20,3))+[1,0,0]]
    poses=np.stack([x,x]);eligible=np.ones(40,dtype=bool)
    labels,stats=spatial_motion_graph(poses,np.array([0.,1.]),eligible,np.ones(40),
        neighbors=10,radius_cap=.1,min_cluster_size=5)
    assert stats['actual_groups']==2 and labels[0]!=labels[20]


def test_direct_graph_uses_motion_to_separate_spatially_near_trajectories():
    rng=np.random.default_rng(1);x=rng.normal(0,.001,(40,3));times=np.array([0.,.4,1.])
    velocity=np.repeat([[.02,0,0],[-.02,0,0]],20,axis=0)
    poses=x[None]+times[:,None,None]*velocity[None]
    labels,stats=spatial_motion_graph(poses,times,np.ones(40,dtype=bool),np.ones(40),
        neighbors=39,radius_factor=100.,radius_cap=1.,motion_radius=.2,min_cluster_size=5)
    assert stats['actual_groups']==2 and labels[0]!=labels[20]
    assert stats['motion_compatible_edges']<stats['candidate_edges']


def test_public_functions_have_no_requested_group_count():
    for function in (density_motion,spatial_motion_graph):
        assert not {'groups','n_clusters','base_labels','base_K'} & set(inspect.signature(function).parameters)
