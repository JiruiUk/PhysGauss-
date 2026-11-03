import os
import torch


def graph_state(scene):
    return [{'neighbors':None if g.neighbors is None else g.neighbors.detach().cpu().clone(),'last':g.last}
        for g in scene.physics.graphs]


def restore_graphs(scene,values):
    if len(values)!=len(scene.physics.graphs):raise ValueError('Physics graph count mismatch')
    for graph,value in zip(scene.physics.graphs,values):
        graph.neighbors=None if value['neighbors'] is None else value['neighbors'].to(scene.centers.device)
        graph.last=value['last']


def save_complete(directory,scene,opt,step,stack,protocol_hash,parent_hash):
    from ..prepare_data import write,sha
    import random
    import numpy as np
    temp=directory/'checkpoint.tmp';path=directory/'checkpoint.pt'
    torch.save({'model':scene.state_dict(),'optimizer':opt.state_dict(),'iteration':step,'camera_stack':list(stack),
        'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
        'rng_cuda':torch.cuda.get_rng_state_all(),'physics_graphs':graph_state(scene),
        'protocol_sha256':protocol_hash,'parent_sha256':parent_hash,'stage':'S3',
        'physical_time_units':'sequence normalized; acquisition seconds unknown'},temp)
    os.replace(temp,path);write(directory/'checkpoint.json',{'sha256':sha(path),'iteration':step,'full_state':True,'physics_graph_cache_saved':True})
