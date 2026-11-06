"""Fixed original-patch denominator diagnostics; parent proxies never GT."""
from pathlib import Path
import argparse
import numpy as np
import torch
from variants.staged_reconstruction.i10_surface_support import runtime as a
from variants.staged_reconstruction.i10_surface_support.structure import graph_inputs
from variants.staged_reconstruction.i10_surface_support.core import covariance, geometry_terms, footprint
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha
from variants.staged_reconstruction.rubber_visibility.worker import native, colors_for
from variants.staged_reconstruction.train_static import camera
from experiments.staged_reconstruction.RubberDuck_surface_support.evaluate import load_models


@torch.no_grad()
def run():
    r=a.install(True);r.verify();torch.set_num_threads(4)
    renderer,models,_=load_models();bank=read(a.REPORT/'surface_bank.json');original=graph_inputs(a.CONFIG)
    edges=original['edges'];result={}
    for label,scene in models.items():
        rows=[]
        for index,frame in enumerate(a.META['frames']):
            assert not r.stopped();active=[q for q in bank['patches'] if str(index) in q['reference']]
            if not active:continue
            dynamic=scene.dynamic_at(frame['time']);cov=covariance(dynamic.get_scaling,dynamic.get_rotation)
            cam=camera(frame,a.META,400)
            for start in range(0,len(active),3):
                batch=active[start:start+3]
                packed=native(renderer,scene,cam,frame['time'],colors_for(scene,[q['rows'] for q in batch]))
                for channel,patch in enumerate(batch):
                    reference=patch['reference'][str(index)];ids=patch['rows']
                    pairs=[(patch['nodes'].index(edges[i]['i']),patch['nodes'].index(edges[i]['j'])) for i in patch['edge_indices']]
                    terms=geometry_terms(dynamic.get_xyz[ids],cov[ids],pairs,reference,a.CONFIG)
                    actual=terms['spectrum'].cpu().numpy();old=np.array(reference['spectrum'])
                    contribution=footprint(packed['render'][channel],reference['uv'])
                    parent=reference['contribution_mean'];known=parent is not None and contribution is not None
                    rows.append(dict(id=patch['id'],frame=index,denominator_fixed=True,
                        thickness_ratio=float(actual[0]/old[0]),tangent_area_ratio=float(np.prod(actual[1:])/np.prod(old[1:])),
                        connection_penalty=float(terms['connection']),thickness_penalty=float(terms['thickness']),
                        area_scale_penalty=float(terms['area_scale']),
                        parent_coverage_known=parent is not None,coverage_known=known,
                        coverage_ratio=float(contribution)/parent if known else None))
        finite=[v for v in rows if v['coverage_known']]
        result[label]=dict(all_original_patch_frames=len(rows),missing_geometry=0,
            known_parent_coverage=sum(v['parent_coverage_known'] for v in rows),
            missing_coverage=sum(v['parent_coverage_known'] and not v['coverage_known'] for v in rows),
            mean_connection_penalty=float(np.mean([v['connection_penalty'] for v in rows])),
            thickness_outside_interval=sum(not .8<=v['thickness_ratio']<=1.25 for v in rows),
            area_outside_interval=sum(not .8<=v['tangent_area_ratio']<=1.25 for v in rows),
            coverage_below_lower=sum(v['coverage_ratio']<.8 for v in finite),
            mean_coverage_ratio=float(np.mean([v['coverage_ratio'] for v in finite])),rows=rows)
        print('L5_STRUCTURAL_DIAGNOSTIC',label,{k:v for k,v in result[label].items() if k!='rows'},flush=True)
    r.verify();write(a.REPORT/'structural_diagnostic.json',dict(passed=True,models=result,
        original_patch_denominator_not_final_active_subset=True,physical_thickness_GT=False,
        parent_reference_not_correct_geometry_claim=True,source_sha256=sha(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--enable-structural-diagnostic',action='store_true')
    if not p.parse_args().enable_structural_diagnostic:raise SystemExit('Diagnostic default off')
    run()
