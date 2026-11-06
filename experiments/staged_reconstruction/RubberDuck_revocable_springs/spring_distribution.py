"""Supplement the preregistered stretch/compression counts; read-only full denominator."""
from pathlib import Path
import argparse
import numpy as np
import torch
from variants.staged_reconstruction.i10_springs import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha
from experiments.staged_reconstruction.RubberDuck_revocable_springs.evaluate import load_models


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-spring-distribution',action='store_true')
    if not p.parse_args().enable_spring_distribution:raise SystemExit('Distribution default off')
    r=a.install(True);r.verify();torch.set_num_threads(4)
    source=read(a.REPORT/'spring_distribution_freeze.json')
    assert sha(Path(__file__))==source['source_sha256']
    renderer,models,_=load_models();bank=read(a.REPORT/'surface_bank.json');graph=read(ROOT/a.CONFIG['graph_bank'])
    result={}
    for name,scene in models.items():
        values=[];rows=[]
        for index,frame in enumerate(a.META['frames']):
            assert not r.stopped();selected=[q for q in bank['patches'] if str(index) in q['reference']]
            if not selected:continue
            xyz=scene.dynamic_at(frame['time']).get_xyz.cpu().numpy().astype(np.float64)
            for q in selected:
                lengths=[]
                for edge in q['edge_indices']:
                    e=graph['edges'][edge];i,j=[graph['patches'][e[k]]['rows'][0] for k in ['i','j']]
                    lengths.append(float(np.linalg.norm(xyz[j]-xyz[i])))
                rest=np.array(q['spring_rest_lengths']);strain=(np.array(lengths)-rest)/rest
                rows.append(dict(id=q['id'],frame=index,lengths=lengths,rest=rest.tolist(),strain=strain.tolist()))
                values.extend(strain.tolist())
        v=np.array(values);assert len(rows)==360 and len(v)==2022 and np.isfinite(v).all()
        result[name]=dict(patch_frames=360,patch_edge_frame_applications=2022,
            positive_stretch=int((v>0).sum()),negative_compression=int((v<0).sum()),exact_zero=int((v==0).sum()),
            quantiles={str(q):float(np.quantile(v,q)) for q in [0,.05,.5,.95,1]},
            sign_not_geometry_error=True,rows=rows)
    r.verify();write(a.REPORT/'spring_distribution.json',dict(passed=True,denominator_initial_not_final_active=True,
        no_optimizer_updates=True,not_independent_geometry_GT=True,source_sha256=sha(Path(__file__)),models=result))
    print('L6_SPRING_DISTRIBUTION_PASS',{n:{k:v for k,v in q.items() if k!='rows'} for n,q in result.items()},flush=True)


if __name__=='__main__':main()
