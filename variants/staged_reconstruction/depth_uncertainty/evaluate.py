"""Independent static-GT calibration audit, never exports GT as training input."""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image,ImageDraw
from scipy.stats import rankdata
from ..prepare_data import ROOT,read,write,sha
from .prepare import BASE,REPORT,RUNS,PARENT,SOURCE,verify,stopped


def describe(values):
    values = np.asarray(values)
    return {'n':int(values.size),'mean':float(values.mean()) if values.size else None,
        'median':float(np.median(values)) if values.size else None,
        'p90':float(np.quantile(values,.9)) if values.size else None}


def color(values,valid,maximum,height,width,uv):
    mapped = np.clip(values/maximum,0,1); colors = cv2.cvtColor(cv2.applyColorMap((mapped*255).astype('uint8')[:,None],cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)[:,0]
    image = np.zeros((height,width,3),np.uint8); image[uv[valid,1],uv[valid,0]] = colors[valid]
    return image


def auc_bad(score,bad):
    n,m = int(bad.sum()),int((~bad).sum())
    if not n or not m: return None
    return float((rankdata(-score)[bad].sum()-n*(n+1)/2)/(n*m))


def main():
    assert not stopped(), 'Saved user stop'
    verify(); assert not (REPORT/'evaluation.json').exists(), 'Preserve completed evaluation'
    receipt = read(REPORT/'inference.json'); protocol = read(BASE/'protocol.json')
    for p,h in receipt['outputs'].items(): assert sha(ROOT/p) == h,p
    for p,h in read(SOURCE/'dataset.json')['files'].items(): assert sha(ROOT/p) == h,p
    results,gates,pictures = {},{},{}
    output = ROOT/'output/staged_reconstruction/S1_depth_uncertainty/v1'; output.mkdir(parents=True,exist_ok=True)
    for case in protocol['synthetic_cases']:
        meta = read(ROOT/f'data/derived/clean_static_synthetic/v2/{case}/train/cameras.json')
        columns = {k:[] for k in ['depth','lower','upper','width','confidence_score','tight','gt','visible','thin','max_reprojection','closer','poses']}
        panels,frames = [],[]
        for i,row in enumerate(meta['frames']):
            assert not stopped(), 'Saved user stop'
            with np.load(RUNS/case/f'view_{i:03d}.npz') as data: saved = {key:data[key].copy() for key in data.files}
            with np.load(PARENT/case/f'view_{i:03d}.npz') as data:
                yy,xx = np.nonzero(data['refined'] > 0); uv = np.c_[xx,yy]
                assert np.array_equal(saved['uv'],uv) and np.array_equal(saved['depth'],data['refined'][yy,xx])
            uv = saved['uv']; yy,xx = uv[:,1],uv[:,0]
            private = ROOT/f'data/derived/clean_static_synthetic/v2/{case}/evaluation_only/gt'/(Path(row['image']).stem+'.npz')
            with np.load(private) as gt:
                target,visible,thin = gt['static_depth'][yy,xx],gt['observed_static_visible'][yy,xx],gt['thin_static_visible'][yy,xx]
            for key in ['depth','lower','upper','width','confidence_score','tight']: columns[key].append(saved[key])
            for key,value in [('gt',target),('visible',visible),('thin',thin),('max_reprojection',saved['maximum_reprojection_px']),
                    ('closer',saved['closer_surface_unknown_count']),('poses',saved['distinct_source_poses'])]: columns[key].append(value)
            frames.append({'image':row['image'],'time':row['time'],'candidates':len(uv),'tight':int(saved['tight'].sum()),
                'available_source_poses_mean':float(saved['distinct_source_poses'].mean())})
            if i in [0,24,72]:
                height,width = meta['height'],meta['width']; on = target > 0; error = abs(saved['depth']-target)
                canvas = Image.new('RGB',(width*4,height+36),'#171923'); draw = ImageDraw.Draw(canvas)
                tight_gt = np.zeros((height,width,3),np.uint8); tight_gt[yy,xx] = [65,65,65]
                good = saved['tight']&on&(error <= .05); bad = saved['tight']&~good
                tight_gt[yy[good],xx[good]] = [40,220,80]; tight_gt[yy[bad],xx[bad]] = [250,50,60]
                values = [('Center abs error 0..0.10',color(error,on,.1,height,width,uv)),
                    ('Evidence score 0..1, not P',color(saved['confidence_score'],np.ones(len(uv),bool),1,height,width,uv)),
                    ('Interval width 0..0.20',color(saved['width'],np.ones(len(uv),bool),.2,height,width,uv)),
                    ('Tight GT: green good, red bad',tight_gt)]
                for j,(name,image) in enumerate(values):
                    canvas.paste(Image.fromarray(image),(j*width,36)); draw.text((j*width+2,2),f'{case} train{i} '+Path(row['image']).stem,fill='white')
                    draw.text((j*width+2,17),name,fill='white')
                panels.append(np.asarray(canvas))
        data = {k:np.concatenate(v) for k,v in columns.items()}; on = data['gt'] > 0
        error = abs(data['depth']-data['gt']); bad = ~on|(error > .05); tight = data['tight']
        inside = on&(data['gt'] >= data['lower'])&(data['gt'] <= data['upper'])
        n = int(tight.sum()); ratio = lambda mask:float(mask.sum()/max(n,1))
        metrics = {'candidates':len(tight),'tight_candidates':n,'tight_candidate_retention':float(tight.mean()),
            'all_GT_interval_coverage':float(inside.mean()),'all_interval_width':describe(data['width']),
            'tight_interval_width':describe(data['width'][tight]),
            'tight_GT_interval_coverage':ratio(tight&inside) if n else None,
            'tight_no_static_or_center_error_gt_005':ratio(tight&bad) if n else None,
            'tight_center_static_depth_error':describe(error[tight&on]),
            'tight_supported_visible_static_retention':float((tight&data['visible']&on).sum()/max((data['visible']&on).sum(),1)),
            'tight_supported_thin_retention':float((tight&data['thin']&on).sum()/max((data['thin']&on).sum(),1)),
            'uncertainty_identifies_bad_AUROC':auc_bad(data['confidence_score'],bad),
            'bad_fraction_all':float(bad.mean()),'bad_fraction_non_tight':float(bad[~tight].mean()) if (~tight).any() else None,
            'possible_closer_surface_ambiguity':int((data['closer'] > 0).sum()),'frames':frames}
        order = np.argsort(-data['confidence_score'],kind='stable'); metrics['ranked_risk_diagnostic_only'] = {}
        for fraction in [.25,.5,.75,1.]:
            ids = order[:max(1,int(len(order)*fraction))]
            metrics['ranked_risk_diagnostic_only'][str(fraction)] = {'candidates':len(ids),'bad_fraction':float(bad[ids].mean()),
                'GT_interval_coverage':float(inside[ids].mean())}
        limit = protocol['evidence_gate_both_cases_required']; failed = []
        checks = [('tight_candidate_retention',limit['tight_candidate_retention_min'],'min'),
            ('tight_supported_visible_static_retention',limit['tight_supported_visible_static_retention_min'],'min'),
            ('tight_supported_thin_retention',limit['tight_supported_thin_retention_min'],'min'),
            ('tight_GT_interval_coverage',limit['tight_GT_interval_coverage_min'],'min'),
            ('tight_no_static_or_center_error_gt_005',limit['tight_no_static_or_center_error_gt_005_max'],'max')]
        for key,bound,direction in checks:
            value = metrics[key]
            if value is None or (value < bound if direction == 'min' else value > bound): failed.append(key)
        mae = metrics['tight_center_static_depth_error']['mean']
        if mae is None or mae > limit['tight_center_static_depth_MAE_max']: failed.append('tight_center_static_depth_MAE')
        if not n: failed.append('empty_support')
        gates[case] = {'passed':not failed,'failed_requirements':failed}; results[case] = metrics
        path = output/f'{case}_source_intervals.png'; assert not path.exists()
        Image.fromarray(np.concatenate(panels)).save(path); pictures[path.relative_to(ROOT).as_posix()] = sha(path)
        print('INTERVAL_GT_AUDIT',case,{k:v for k,v in metrics.items() if k not in ['frames']},gates[case],flush=True)
    for p,h in receipt['outputs'].items(): assert sha(ROOT/p) == h,p
    verify(); write(REPORT/'evaluation.json',{'results':results,'images':pictures,'independent_GT_evaluation':True,
        'GT_not_used_for_inference_or_selection':True,'Gaussian_training':False,'inference_sha256':sha(REPORT/'inference.json')})
    write(REPORT/'gate.json',{'both_cases_evidence_passed':all(g['passed'] for g in gates.values()),'cases':gates,
        'Gaussian_training_started':False,'real_training_authorized':False,
        'next_requires_separate_paired_Gaussian_protocol_and_native_recovery':True})
    print('INTERVAL_AUDIT_COMPLETED',gates,flush=True)


if __name__ == '__main__': main()
