"""Only aggregate sufficient statistics leave evaluator memory."""
import numpy as np


def distribution(values, difference=False):
    a = np.asarray(values,dtype=np.float64).ravel(); a = a[np.isfinite(a)]
    lo,hi = (-1.,1.) if difference else (0.,1.)
    counts = np.histogram(a,bins=100,range=(lo,hi))[0]
    assert counts.sum() == len(a)
    return {'n':len(a),'sum':float(a.sum()),'sum_sq':float(a@a),'range':[lo,hi],
            'histogram':counts.tolist()}


def summarize(costs, fit, accepted, mask):
    c,f,a = costs[:,:,mask],fit[:,mask],accepted[:,mask]
    out = {'n':int(mask.sum()),'continuous_accepted':int(a[0].sum()),'grid_accepted':int(a[1].sum()),
           'unit_source_costs':(c == 1).sum((1,2)).tolist(),
           'fit_nonworse':[(int((f[i] <= f[0]+1e-6).sum())) for i in [1,2]],
           'held_nonworse':[(int((c[i,4:] <= c[0,4:]+1e-6).all(0).sum())) for i in [1,2]],
           'metrics':{}}
    for h,name in enumerate(['D1','GT','gridGT']):
        for key,v in [('fit',f[h]),('held1',c[h,4]),('held2',c[h,5]),('six_sources',c[h])]:
            out['metrics'][name+'_'+key] = distribution(v)
        if h:
            for key,v in [('fit',f[h]-f[0]),('held1',c[h,4]-c[0,4]),('held2',c[h,5]-c[0,5])]:
                out['metrics'][name+'_minus_D1_'+key] = distribution(v,True)
    return out


def groups(single, visibility, error, thin):
    n = len(error); all6 = visibility.all(0)
    enough = (visibility[:4].sum(0) >= 3)&visibility[4:].all(0)
    masks = {'all':np.ones(n,bool),'single_face':single,'mixed_edge_missing':~single,
             'all6_visible_subset':all6,'fit3_bothheld_visible_inclusive':enough,
             'other_visibility':~enough,'visible_thin':thin}
    es = [error <= .02,(error > .02)&(error <= .05),error > .05]
    for en,e in zip(['le02','02to05','gt05'],es):
        masks['error_'+en] = e
        for fn,f in [('single',single),('mixed',~single)]:
            for vn,v in [('all6',all6),('enough_notall6',enough&~all6),('other',~enough)]:
                masks[fn+'/'+vn+'/'+en] = f&v&e
    return masks


def merge(rows):
    """Sum histograms; no pooled privileged pixel arrays are saved."""
    import copy
    out = copy.deepcopy(rows[0])
    for r in rows[1:]:
        for name,g in r.items():
            dest = out[name]
            for key in ['n','continuous_accepted','grid_accepted']:
                dest[key] += g[key]
            for key in ['unit_source_costs','fit_nonworse','held_nonworse']:
                dest[key] = [a+b for a,b in zip(dest[key],g[key])]
            for key,d in g['metrics'].items():
                x = dest['metrics'][key]
                for field in ['n','sum','sum_sq']: x[field] += d[field]
                x['histogram'] = [a+b for a,b in zip(x['histogram'],d['histogram'])]
    return out


def compact(groups):
    output = {}
    for name,g in groups.items():
        row = {k:v for k,v in g.items() if k != 'metrics'}; row['metrics'] = {}
        for key,d in g['metrics'].items():
            n = d['n']; lo,hi = d['range']; hist = np.asarray(d['histogram'])
            q = [float(lo+(min(np.searchsorted(hist.cumsum(),p*n,side='right'),99)+.5)*(hi-lo)/100) for p in [.1,.5,.9]] if n else None
            row['metrics'][key] = {'n':n,'mean':d['sum']/n if n else None,'p10_p50_p90_histogram_midpoints':q}
        output[name] = row
    return output
