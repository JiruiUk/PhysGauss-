"""Native flags-off repeat envelope plus enabled checkpoint geometry invariance."""
import os
import sys
import torch
from .run import ROOT, REPORT, RUNS, DATA, engine
from .prepare import prepare


def main():
    torch.set_num_threads(4);sys.path.insert(0,str(prepare()))
    from variants.association_training.evaluate import load,render_call
    from variants.shape_coherence.observations import camera
    from variants.flow_comparison.temporal import warp_at
    from variants.color_repair.color import _read
    from .probe import geometry
    from .render import pool_for
    meta=engine.read_json(DATA/'transforms_train.json')
    frames={int(f['file_path'].split('/')[-1]):f for f in meta['frames']}
    outputs={};proofs={}
    for name in ['original','original_repeat','flags_off','enabled_resume']:
        path=RUNS/'_smoke'/name/'rubber_duck'
        os.environ['PIDG_METRIC_CONFIG']=str(path/'metric_config.json');_read.cache_clear()
        g,d,proof=load(path);proofs[name]=proof
        values=[]
        with torch.no_grad():
            for n in [2,86,188]:
                w=warp_at(d,g.get_xyz.detach(),(n-1)/199)
                pkg=render_call(camera(frames[n],meta,1600),g,w)
                values.append({k:pkg[k].detach().cpu() for k in ['render','alpha']})
            if name=='enabled_resume':
                ids=pool_for(g).ids;previous=None
                for t in [0.,.25,.5,.75,1.]:
                    this=[v[ids] for v in geometry(g,warp_at(d,g.get_xyz.detach(),t))]
                    if previous is not None:assert all(torch.equal(a,b) for a,b in zip(previous,this))
                    previous=this
        outputs[name]=values;del g,d;torch.cuda.empty_cache()
    comparisons={}
    for name in ['original_repeat','flags_off']:
        comparisons[name]={}
        for channel in ['render','alpha']:
            mse=sum(float((a[channel].double()-b[channel].double()).square().mean()) for a,b in zip(outputs[name],outputs['original']))/3
            comparisons[name][channel+'_RMS']=mse**.5
    for channel in ['render','alpha']:
        assert comparisons['flags_off'][channel+'_RMS']<=3*comparisons['original_repeat'][channel+'_RMS']+1e-7
    engine.write_json(REPORT/'native_check.json',{'passed':True,'native_resolution':[1600,900],
        'training_frames':[2,86,188],'full_states':proofs,'comparisons':comparisons,
        'gate':'RMS <= 3 x unchanged original-repeat RMS + 1e-7',
        'enabled_restored_static_center_covariance_time_invariant':True,'new_view_GT_claim':False})
    print('Native off-parity and restored static invariance passed',comparisons,flush=True)


if __name__=='__main__':main()
