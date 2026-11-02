from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.short_clip_diagnostic.image_camera import REPORT,OUT,read,write,sha
from output.short_clip_diagnostic.model_check import CACHE
from output.surface_correspondence.audit import center_inside
from variants.surface_correspondence.model import particle_support,Config


def main():
    r=read(REPORT/'model_results.json');image=read(REPORT/'image_camera.json');config=Config(enabled=True)
    with np.load(ROOT/'output/surface_support/image_tracks.npz') as z:components={int(n):c for n,c in zip(z['frames'],z['components'])}
    numbers=[x['frame'] for x in r['frames']];data={}
    for n in numbers:
        with np.load(CACHE/f'frame{n:04d}.npz') as z:data[n]={k:z[k] for k in z.files}
    from output.short_clip_diagnostic.model_check import flow_at_centers
    from output.short_clip_diagnostic.image_camera import FLOW
    from variants.flow_comparison.geometry import sample,forward_backward_valid
    import torch
    torch.set_num_threads(4)
    manifest=read(FLOW/'manifest.json');lookup={(p['source'],p['target']):p for p in manifest['pairs']};rows=[]
    for clip in r['clips']:
        frames=clip['frames'];masses=[]
        for i,n in enumerate(frames):
            if i<3:continue
            previous=frames[i-1];target=data[n];source=data[previous];pair=lookup[(previous,n)]
            with np.load(FLOW/pair['file']) as z:f=torch.tensor(z['forward'],dtype=torch.float32);b=torch.tensor(z['backward'],dtype=torch.float32)
            p=torch.tensor(source['means']);q=torch.tensor(target['means']);valid=sample(forward_backward_valid(f,b).float(),p[:,None])[:,0]>.99
            valid&=(p[:,0]>=0)&(p[:,0]<=399)&(p[:,1]>=0)&(p[:,1]<=224)
            error=torch.linalg.vector_norm(q-p-sample(f,p[:,None])[:,0],dim=-1).numpy();error[~valid.numpy()]=np.inf
            point_ids=set();rawmass=0.;pointmass=0.
            for bank in clip['banks']:
                if not bank['qualified'] or bank['name'].endswith('blue'):continue
                bb=next(x for x in r['banks'] if x['name']==bank['name'])
                with np.load(ROOT/bb['path']) as z:ix=z['kernel_ids']
                current=next(x for x in r['bank_frames'] if x['clip']==clip['clip'] and x['frame']==n and x['bank']==bank['name'])['region']
                old=next(x for x in r['bank_frames'] if x['clip']==clip['clip'] and x['frame']==previous and x['bank']==bank['name'])['region']
                inside=center_inside(target['means'][ix],target['depth'][ix],target['cov'][ix],components[n]==current)
                was_inside=center_inside(source['means'][ix],source['depth'][ix],source['cov'][ix],components[previous]==old)
                good=particle_support(target['mass'][ix,current],target['mass'][ix,5],inside,error[ix],config)
                good&=particle_support(source['mass'][ix,old],source['mass'][ix,5],was_inside,None,config)
                rawmass+=float(target['mass'][ix,current].sum())
                for index in ix[good]:
                    assert int(index) not in point_ids;point_ids.add(int(index));pointmass+=float(target['mass'][index,current])
            denominator=float(target['alpha'][components[n]>=0].sum())
            masses.append({'frame':n,'raw_bank_target_alpha_coverage':rawmass/denominator,'point_only_alpha_coverage':pointmass/denominator,
                 'original_cohort_plus_point_alpha_coverage':next(x['accepted_duck_alpha_coverage'] for x in clip['coverage'] if x['frame']==n)})
        rows.append({'clip':clip['clip'],'frames':masses,'raw_bank_mean_alpha_coverage':float(np.mean([x['raw_bank_target_alpha_coverage'] for x in masses])),
          'point_only_mean_alpha_coverage':float(np.mean([x['point_only_alpha_coverage'] for x in masses])),
          'original_joint_mean_alpha_coverage':clip['mean_forward_accepted_duck_alpha_coverage']})
    write(REPORT/'gate_analysis.json',{'clips':rows,'thresholds_and_banks_unchanged':True,'new_optimizer_updates':0,
       'posthoc_interpretability_check':True,'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}})
    print('POINT VERSUS COHORT GATES',[(x['clip'],x['original_joint_mean_alpha_coverage'],x['point_only_mean_alpha_coverage'],x['raw_bank_mean_alpha_coverage']) for x in rows],flush=True)


if __name__=='__main__':main()
