"""Independent CPU arithmetic, bank/gate, image, and protected-model receipts."""
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.short_clip_diagnostic.image_camera import REPORT,OUT,TRAIN,FLOW,read,write,sha,HYPOTHESES,image
from output.short_clip_diagnostic.camera_math import fundamental,w2c,intrinsic,project
from output.surface_correspondence.audit import center_inside
from variants.flow_comparison.geometry import sample,forward_backward_valid


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def point_good(mass,total,inside,error=None):
    v=(mass>=.02)&(mass/np.maximum(total,1e-20)>=.95)&inside
    if error is not None:v&=np.isfinite(error)&(error<=2)
    return v


def cohort_good(mass,total,inside,error):
    a=float(mass.sum());b=float(total.sum())
    return a>=.2 and b>0 and a/b>=.95 and np.sum(total*inside)/b>=.9 and np.sum(mass*(np.isfinite(error)&(error<=2)))/a>=.9


def main():
    torch.set_num_threads(4)
    for file in ['input_lock.json','model_input_lock.json']:
        assert all(sha(ROOT/p)==v for p,v in read(REPORT/file).items())
    assert all(sha(ROOT/p)==v for p,v in read(REPORT/'cache_manifest.json').items())
    for file in ['resolution_control.json','gate_analysis.json']:
        assert all(sha(ROOT/p)==v for p,v in read(REPORT/file)['source_sha256'].items())
    model=read(REPORT/'model_results.json');cam=read(REPORT/'image_camera.json');native=read(REPORT/'resolution_control.json');secondary=read(REPORT/'gate_analysis.json')
    from reproduction.scripts.training_checkpoint import load_state
    path=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck/checkpoints/resume.pt';assert sha(path)==model['checkpoint_sha256']
    state=load_state(path);canonical=state['gaussian_parameters']['_xyz'].numpy()
    assert state['gaussian_scalars']['active_sh_degree']==3 and not state['pipe']['compute_cov3D_python'] and not state['pipe']['convert_SHs_python']
    assert state['dataset']['resolution']==1600;del state
    meta=read(TRAIN/'transforms_train.json');lookup={int(Path(f['file_path']).stem):f for f in meta['frames']}
    source=arrays(ROOT/'output/surface_support/image_tracks.npz');components={int(n):c for n,c in zip(source['frames'],source['components'])}
    cache=ROOT/'runs/pidg_short_clip_diagnostic/cache';numbers=[r['frame'] for r in model['frames']]
    assert len(numbers)==24 and len(set(numbers))==24
    data={n:arrays(cache/f'frame{n:04d}.npz') for n in numbers};high={n:arrays(cache/f'native{n:04d}.npz') for n in numbers}
    for n in numbers:
        _,blue=image(lookup[n]);masks=[*(components[n]==j for j in range(4)),blue,np.ones((225,400),bool)]
        for d in [data[n],high[n]]:
            assert np.isfinite(d['mass']).all() and (d['mass']>=0).all()
            for j,mask in enumerate(masks):
                expected=float(d['alpha'][mask].sum());actual=float(d['mass'][:,j].sum())
                assert abs(actual-expected)/max(expected,1)<5e-4
        xy,_=project(data[n]['centers'],w2c(lookup[n]['transform_matrix']),intrinsic(meta['camera_angle_x']))
        good=data[n]['depth']>.2;assert np.max(np.abs(xy[good]-data[n]['means'][good]))<.003
    # Recalculate saved camera residuals directly, not from model geometry.
    points=arrays(OUT/'image_correspondences.npz');cv=read(REPORT/'camera_values.json')
    for pair in cam['pairs']:
        a,b=pair['source'],pair['target']
        for method in ['raw_blue','LK_blue']:
            x=points[f'{a}_{b}_{method}_source'];y=points[f'{a}_{b}_{method}_target'];xx=np.c_[x,np.ones(len(x))];yy=np.c_[y,np.ones(len(y))]
            for pose,k in HYPOTHESES:
                key=pose+'__'+k;aa=cv[str(a)][key];bb=cv[str(b)][key]
                f=fundamental(np.array(aa['w2c']),np.array(bb['w2c']),np.array(aa['K']),np.array(bb['K']))
                fx=xx@f.T;fy=yy@f;distance=np.abs(np.sum(yy*fx,1))/np.sqrt(np.sum(fx[:,:2]**2,1)+np.sum(fy[:,:2]**2,1))
                stats=pair['camera_scores'][method][key];assert stats['n']==len(x)
                if len(x):assert abs(float(np.median(distance))-stats['median'])<1e-10
    flow_lookup={(p['source'],p['target']):p for p in read(FLOW/'manifest.json')['pairs']};errors={}
    for a,b in sorted({(a,b) for c in model['clips'] for a,b in zip(c['frames'][:-1],c['frames'][1:])}):
        raw=arrays(FLOW/flow_lookup[(a,b)]['file']);f=torch.tensor(raw['forward'],dtype=torch.float32);back=torch.tensor(raw['backward'],dtype=torch.float32)
        p=torch.tensor(data[a]['means']);q=torch.tensor(data[b]['means']);valid=sample(forward_backward_valid(f,back).float(),p[:,None])[:,0]>.99
        valid&=(p[:,0]>=0)&(p[:,0]<=399)&(p[:,1]>=0)&(p[:,1]<=224)
        e=torch.linalg.vector_norm(q-p-sample(f,p[:,None])[:,0],dim=-1).numpy();e[~valid.numpy()]=np.inf;errors[(a,b)]=e
    assignments=arrays(OUT/'assignments.npz');checks=[]
    # Both raster resolutions use the same source/target masks and center test.
    for clip in model['clips']:
        frames=clip['frames'];all_regions={(r['bank'],r['frame']):r['region'] for r in model['bank_frames'] if r['clip']==clip['clip']}
        sets={'400':[],'same400_banks':[],'native_seed_banks':[]}
        for bb in clip['banks']:
            if not bb['qualified']:continue
            bank=next(b for b in model['banks'] if b['name']==bb['name']);assert sha(ROOT/bank['path'])==bank['sha256'];z=arrays(ROOT/bank['path']);ix=z['kernel_ids']
            assert np.array_equal(z['canonical'],canonical[ix]);masks=[]
            for index,n in enumerate(frames[:3]):
                c=all_regions[(bb['name'],n)];mask=components[n]==c if c<4 else image(lookup[n])[1]
                inside=center_inside(data[n]['means'],data[n]['depth'],data[n]['cov'],mask)
                masks.append(point_good(data[n]['mass'][:,c],data[n]['mass'][:,5],inside,None if index==0 else errors[(frames[index-1],n)]))
            assert np.array_equal(np.flatnonzero(np.logical_and.reduce(masks)),ix)
            if bb['name'].endswith('blue'):continue
            sets['400'].append((bb['name'],ix));sets['same400_banks'].append((bb['name'],ix))
            nb=next(x for x in native['native_seed_banks'] if x['clip']==clip['clip'] and x['name']==bb['name'])
            masks=[]
            for index,n in enumerate(frames[:3]):
                c=all_regions[(bb['name'],n)];inside=center_inside(data[n]['means'],data[n]['depth'],data[n]['cov'],components[n]==c)
                masks.append(point_good(high[n]['mass'][:,c],high[n]['mass'][:,5],inside,None if index==0 else errors[(frames[index-1],n)]))
            expected=np.flatnonzero(np.logical_and.reduce(masks));assert nb['qualified']==(len(expected)>=20)
            if nb['qualified']:
                assert sha(ROOT/nb['path'])==nb['sha256'];zz=arrays(ROOT/nb['path']);assert np.array_equal(expected,zz['kernel_ids'])
                sets['native_seed_banks'].append((bb['name'],expected))
        for mode,banks in sets.items():
            contributions=[]
            for i,n in enumerate(frames):
                if i<3:
                    if mode=='400':assert (assignments[clip['clip']][i]<0).all()
                    continue
                previous=frames[i-1];values=data if mode=='400' else high;assigned=np.full(len(canonical),-1,int)
                for name,ix in banks:
                    c=all_regions[(name,n)];pc=all_regions[(name,previous)]
                    inside=center_inside(data[n]['means'][ix],data[n]['depth'][ix],data[n]['cov'][ix],components[n]==c)
                    was=center_inside(data[previous]['means'][ix],data[previous]['depth'][ix],data[previous]['cov'][ix],components[previous]==pc)
                    m=values[n]['mass'][ix,c];total=values[n]['mass'][ix,5];e=errors[(previous,n)][ix]
                    good=point_good(m,total,inside,e)&point_good(values[previous]['mass'][ix,pc],values[previous]['mass'][ix,5],was)
                    if not cohort_good(m,total,inside,e):good[:]=False
                    assigned[ix[good]]=c
                selected=np.where(assigned>=0)[0];coverage=float(values[n]['mass'][selected,assigned[selected]].sum()/values[n]['alpha'][components[n]>=0].sum());contributions.append(coverage)
                if mode=='400':assert np.array_equal(assigned,assignments[clip['clip']][i])
            expected=clip['mean_forward_accepted_duck_alpha_coverage'] if mode=='400' else next(c for c in native['full_clip_comparisons'] if c['clip']==clip['clip'])['comparison'][mode]['mean_forward_alpha_coverage']
            assert abs(np.mean(contributions)-expected)<1e-6
            checks.append({'clip':clip['clip'],'mode':mode,'mean_forward_alpha_coverage':float(np.mean(contributions))})
    # All24 pixel concatenations; native references match the actual loader.
    native_png=[]
    for n in numbers:
        with Image.open(OUT/'gt_render'/f'frame{n:04d}.png') as im:a=np.array(im.convert('RGB'));assert im.size==(800,225)
        gt,_=image(lookup[n]);assert np.array_equal(a[:,:400],gt) and np.array_equal(a[:,400:],np.rint(data[n]['rgb']*255).astype('uint8'))
        with Image.open(TRAIN/(lookup[n]['file_path']+'.png')) as im:rgba=np.array(im.convert('RGBA'))/255
        expected=np.array(Image.fromarray((rgba[...,:3]*rgba[...,3:4]*255).astype('uint8')).resize((1600,900),Image.Resampling.BICUBIC))
        with Image.open(OUT/'native_gt_render'/f'frame{n:04d}.png') as im:a=np.array(im.convert('RGB'));assert im.size==(3200,900)
        assert np.array_equal(a[:,:1600],expected);mse=np.mean(((a[:,1600:].astype(float)-expected)/255)**2);png_psnr=float(-10*np.log10(mse))
        original=next(r['native1600_vs_trainingGT_PSNR'] for r in native['RGB'] if r['frame']==n)
        assert abs(np.sqrt(mse)-10**(-original/20))<=.5/255+1e-7
        native_png.append({'frame':n,'independent_native_PNG_PSNR':png_psnr})
    for row in model['counterfactuals']:
        assert row['counterfactual_is_not_GT']
        assert all(np.isfinite(row[k]) and row[k]>=0 for k in ['normal_scene_own_mass','isolated_actual_mass','isolated_seed_footprint_mass','isolated_seed_world_center_mass'])
        assert row['isolated_actual_mass']+.02>=row['normal_scene_own_mass']
    from output.c1_flow_ablation.reuse_control import control_hashes
    assert control_hashes()==read(ROOT/'experiments/c1_flow_ablation/reuse_control_plan.json')['historical_control_sha256']
    assert read(REPORT/'manual_review.json')['review_complete']
    files={str(p.relative_to(ROOT)):sha(p) for base in [REPORT,OUT] for p in base.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.npz','.png','.html','.md'] and p.name!='verification.json'}
    write(REPORT/'verification.json',{'passed':True,'old_source_data_checkpoint_SHA_unchanged':True,'historical_C1_protection_passed':True,
        'full_checkpoint_SH_and_pipeline_settings_verified':True,'actual_training_width':1600,'unique_training_RGB_frames':24,'unique_camera_pairs':21,
        'epipolar_residuals_independently_recomputed':True,'fixed_seed_intersections_recomputed':True,'native_reseed_intersections_recomputed':True,
        'all_source_target_point_unknowns_recomputed':True,'coverage_checks':checks,'all48_GT_Render_PNGs_pixel_checked':True,'native_PNG_scores':native_png,
        'model_optimizer_steps':0,'physical_3D_or_identity_GT_claim':False,'native_controls_preserve_400_masks':True,'artifact_sha256':files})
    print('FOUR-CLIP INDEPENDENT VERIFICATION PASSED',len(checks),len(native_png),flush=True)


if __name__=='__main__':main()
