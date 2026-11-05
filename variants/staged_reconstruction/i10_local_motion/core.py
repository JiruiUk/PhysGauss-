"""Persistent visible observations and local corrections, without extrapolation."""
import numpy as np


def support(evidence, parent_xyz):
    parent_xyz=np.asarray(parent_xyz)
    observations={};edges={};accepted=0;qualified=0
    for track in evidence['rows']:
        binding=track['binding']
        if not track['accepted']:continue
        assert binding['accepted'] and track['reciprocal']['passed'] and track['visibility']['passed']
        accepted+=1;anchor=binding['row'];frames=track['visibility_by_row'][str(anchor)]
        graph=track['neighborhood'];qualified+=int(graph['qualified'])
        for local,uv in enumerate(track['uv']):
            if not frames[local]['passed']:continue
            frame=track['start']+local
            observations.setdefault((frame,anchor),[]).append(uv)
            if not graph['qualified']:continue
            for neighbor in graph['rows']:
                i,j=sorted([anchor,neighbor]);assert i!=j
                key=(frame,i,j);offset=(parent_xyz[frame,j]-parent_xyz[frame,i]).tolist()
                if key in edges:assert edges[key]==offset
                edges[key]=offset
    obs={};pairs={}
    for (frame,row),uvs in sorted(observations.items()):
        obs.setdefault(frame,[]).append((row,np.mean(uvs,axis=0).tolist()))
    for (frame,i,j),offset in sorted(edges.items()):pairs.setdefault(frame,[]).append((i,j,offset))
    summary=dict(accepted_tracks=accepted,qualified_graph_tracks=qualified,
                 observations=sum(map(len,obs.values())),supported_track_frames=len(obs),
                 pairs=sum(map(len,pairs.values())),supported_pair_frames=len(pairs),
                 track_counts_by_frame={str(i):len(obs.get(i,[])) for i in range(len(parent_xyz))},
                 pair_counts_by_frame={str(i):len(pairs.get(i,[])) for i in range(len(parent_xyz))},
                 original_global_coverage_not_promoted=True)
    return obs,pairs,summary


def add_local(base,world,entries,enabled=False,weight=.01,scale=.05):
    if not enabled or not entries:return base,base*0
    import torch
    i=torch.tensor([e[0] for e in entries],device=world.device,dtype=torch.long)
    j=torch.tensor([e[1] for e in entries],device=world.device,dtype=torch.long)
    reference=world.new_tensor([e[2] for e in entries])
    error=((world[j]-world[i])-reference)/scale
    term=weight*(torch.sqrt(1+error.square().sum(-1))-1).mean()
    return base+term,term
