"""Motion-only grouping must not depend on color or absolute translation."""
import numpy as np
import pytest

from variants.object_centers.motion_grouping import (
    motion_features, cluster_motion, persistent_components,
)


@pytest.mark.parametrize('mode',['displacement','displacement_velocity'])
def test_features_ignore_independent_constant_kernel_offsets(mode):
    rng=np.random.default_rng(2)
    poses=rng.normal(size=(4,30,3));times=np.array([0.,.1,.6,1.])
    shifts=rng.normal(size=(1,30,3))*10
    assert np.allclose(motion_features(poses,times,mode),
                       motion_features(poses+shifts,times,mode),atol=1e-6)


def test_velocity_uses_actual_nonuniform_time_intervals():
    times=np.array([0.,.1,.6,1.]);v=np.array([[1.,2.,0.],[-2.,0.,1.]])
    poses=times[:,None,None]*v[None]
    features=motion_features(poses,times,'displacement_velocity')
    # Velocity block is interval-weighted, with a common amplitude scale.
    blocks=features[:,len(times)*3:].reshape(2,len(times)-1,3)
    raw=blocks/np.sqrt(np.diff(times))[None,:,None]
    assert np.allclose(raw[:,0],raw[:,1]) and np.allclose(raw[:,1],raw[:,2])


def test_pure_motion_separates_coincident_opposite_trajectories_deterministically():
    rng=np.random.default_rng(0);n=60;times=np.array([0.,.2,.7,1.])
    x=rng.normal(0,.01,(2*n,3));v=np.repeat([[0.,1.,0.],[0.,-1.,0.]],n,axis=0)
    poses=x[None]+times[:,None,None]*v[None];eligible=np.ones(2*n,dtype=bool);eligible[-1]=False
    a=cluster_motion(poses,times,eligible,np.ones(2*n),2,'displacement')
    b=cluster_motion(poses,times,eligible,np.ones(2*n),2,'displacement')
    assert np.array_equal(a,b) and a[-1]==-1
    assert len(set(a[:n]))==len(set(a[n:-1]))==1 and a[0]!=a[n]


def test_connectivity_splits_separate_static_bodies_and_drops_small_fragment():
    rng=np.random.default_rng(0)
    x=np.r_[rng.normal(0,.01,(20,3)),rng.normal(0,.01,(20,3))+[1,0,0],[[4,0,0]]]
    poses=np.stack([x,x]);base=np.zeros(len(x),dtype=np.int32)
    labels,stats=persistent_components(poses,base,np.ones(len(x)),neighbors=6,
        radius_cap=.15,min_points=5)
    assert len(set(labels[:20]))==len(set(labels[20:40]))==1
    assert labels[0]!=labels[20] and labels[-1]==-1
    assert stats['kept_components']==2 and stats['discarded_kernels']==1


def test_brief_initial_contact_does_not_connect_separating_objects():
    rng=np.random.default_rng(8);n=20
    x=rng.normal(0,.002,(2*n,3));times=np.array([0.,.3,.6,1.])
    poses=np.repeat(x[None],4,axis=0)
    poses[:,n:,0]+=times[:,None]
    labels,stats=persistent_components(poses,np.zeros(2*n,dtype=np.int32),np.ones(2*n),
        neighbors=39,radius_factor=100.,radius_cap=.05,persistence=.8,min_points=5)
    assert labels[0]!=labels[n] and (labels>=0).all()
    assert stats['kept_components']==2


@pytest.mark.parametrize('times',[[0.,0.],[1.,0.],[0.,float('nan')]])
def test_invalid_time_knots_rejected(times):
    with pytest.raises(ValueError):motion_features(np.zeros((2,4,3)),np.array(times),'displacement')
