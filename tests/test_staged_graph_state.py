from types import SimpleNamespace
import torch
from variants.flow_comparison.local_momentum import Graph
from variants.staged_reconstruction.state_recovery.runtime import graph_state,restore_graphs


def test_restored_cached_graph_keeps_same_neighbors_and_refresh_clock():
    xyz=torch.randn(48,3);g=Graph();g.sample(xyz,10,count=4,k=5)
    scene=SimpleNamespace(physics=SimpleNamespace(graphs=[g]),centers=torch.zeros(1))
    saved=graph_state(scene);other=Graph()
    copy=SimpleNamespace(physics=SimpleNamespace(graphs=[other]),centers=torch.zeros(1))
    restore_graphs(copy,saved)
    changed=xyz.flip(0)
    a=g.sample(changed,80,count=4,k=5);b=other.sample(changed,80,count=4,k=5)
    assert g.last==other.last==10 and all(torch.equal(x,y) for x,y in zip(a,b))
    saved[0]['neighbors'].zero_()
    assert not torch.equal(g.neighbors,saved[0]['neighbors'])


def test_empty_graph_cache_remains_empty_until_first_sample():
    scene=SimpleNamespace(physics=SimpleNamespace(graphs=[Graph()]),centers=torch.zeros(1))
    values=graph_state(scene);restore_graphs(scene,values)
    assert scene.physics.graphs[0].neighbors is None and scene.physics.graphs[0].last==-10000


def test_current_world_neighbors_follow_moved_geometry_and_never_include_self():
    from variants.staged_reconstruction.state_recovery.physics import world_neighborhood
    xyz=torch.randn(80,3);first,anchors,neighbors=world_neighborhood(xyz,20,count=4,k=12)
    distances=torch.cdist(xyz[anchors],xyz);distances[torch.arange(4),anchors]=float('inf')
    assert torch.equal(neighbors,distances.argsort(1)[:,:12])
    assert not bool((neighbors==anchors[:,None]).any()) and torch.isin(neighbors,first).all()
