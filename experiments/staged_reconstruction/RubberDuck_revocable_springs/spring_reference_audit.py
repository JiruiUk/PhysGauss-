"""Independent saved-reference and all360-frame spring arithmetic replay, no updates."""
from pathlib import Path
import argparse
import numpy as np
import torch
from variants.staged_reconstruction.i10_springs import runtime as a
from variants.staged_reconstruction.i10_tracking.io import ROOT,read,write,sha
from experiments.staged_reconstruction.RubberDuck_revocable_springs.evaluate import load_models


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-spring-audit',action='store_true')
    if not parser.parse_args().enable_spring_audit:raise SystemExit('Spring audit default off')
    r=a.install(True);r.verify();torch.set_num_threads(4)
    p=read(a.BASE/'protocol.json');bank=read(a.REPORT/'surface_bank.json')
    freeze=read(a.REPORT/'spring_evaluation_freeze.json')
    assert freeze['source_sha256']==sha(Path(__file__))
    original=read(ROOT/p['surface_reference_parent']);assert bank['fixed_stiffness']==p['shared_stiffness']==1.
    assert len(bank['patches'])==len(original['patches'])==60
    graph=read(ROOT/p['graph_bank']);edges=graph['edges'];allrest=[]
    for q,old in zip(bank['patches'],original['patches']):
        assert {k:v for k,v in q.items() if k!='spring_rest_lengths'}==old
        values=[old['reference'][f]['edge_lengths'] for f in sorted(old['reference'],key=int)]
        assert np.array_equal(np.median(values,axis=0),q['spring_rest_lengths'])
        allrest.extend(q['spring_rest_lengths'])
    renderer,models,_=load_models(); saved=read(a.REPORT/'structural_diagnostic.json')['models'];result={}
    for label,scene in models.items():
        rows=[]
        for index,frame in enumerate(a.META['frames']):
            assert not r.stopped()
            selected=[q for q in bank['patches'] if str(index) in q['reference']]
            if not selected:continue
            xyz=scene.dynamic_at(frame['time']).get_xyz.cpu().numpy()
            for q in selected:
                lengths=np.array([np.linalg.norm(xyz[graph['patches'][edges[i]['j']]['rows'][0]].astype(np.float64)-xyz[graph['patches'][edges[i]['i']]['rows'][0]].astype(np.float64)) for i in q['edge_indices']])
                rest=np.array(q['spring_rest_lengths'],dtype=np.float32).astype(np.float64)
                strain=(lengths-rest)/rest;energy=float(np.mean(np.sqrt(1+strain**2)-1))
                rows.append(dict(id=q['id'],frame=index,energy=energy,mean_abs_strain=float(np.abs(strain).mean())))
        assert len(rows)==len(saved[label]['rows'])==360
        for x,y in zip(rows,saved[label]['rows']):
            assert (x['id'],x['frame'])==(y['id'],y['frame'])
            assert np.isclose(x['energy'],y['spring_energy'],atol=3e-7,rtol=1e-4)
            assert np.isclose(x['mean_abs_strain'],y['mean_abs_strain'],atol=3e-7,rtol=1e-4)
        result[label]=dict(all360_replayed=True,mean_energy=float(np.mean([v['energy'] for v in rows])),mean_abs_strain=float(np.mean([v['mean_abs_strain'] for v in rows])))
    r.verify();write(a.REPORT/'spring_reference_audit.json',dict(passed=True,all337_patch_edges_fixed_fit_median_exact=len(allrest),
        no_GT_material_rest=True,no_optimizer_updates=True,all_original_patch_frames=True,models=result,source_sha256=sha(Path(__file__))))
    print('L6_INDEPENDENT_SPRING_REFERENCE_AND_ARITHMETIC_PASS',result,flush=True)


if __name__=='__main__':main()
