"""One opt-in native visibility audit; no optimizer calls or physical labels."""
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
import torch
from ..rubber_joint import runtime as old
from ..prepare_data import ROOT,DATA,read,sha
from ..train_static import camera,PIPE
from ..stage2.model import compose
from .core import select_ids,packing_colors,weighted_observation,metadata_census

BASE=ROOT/'experiments/staged_reconstruction/RubberDuck_visibility_audit'
REPORT=BASE/'v1'
RUNS=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_visibility_audit/v1'
OUT=ROOT/'output/staged_reconstruction/RubberDuck_visibility_audit/v1'


def stopped():
    return old.stopped() or any(x.exists() for x in [RUNS/'stop.request',RUNS.parent/'stop.request'])


def verify():
    old.verify()
    for path in [BASE/'preregistration.json',REPORT/'source_freeze.json']:
        d=read(path);d=d.get('files',d)
        for name,h in d.items():assert sha(ROOT/name)==h,name


def load(name):
    p=read(BASE/'protocol.json');row=p['models'][name]
    assert sha(ROOT/row['path'])==row['sha256']
    renderer,scene,opts,teacher,parent=old.load(name if name!='parent6k' else 'frozen_static')
    state=parent if name=='parent6k' else old.restore(ROOT/row['path'],scene,opts)
    step=0 if name=='parent6k' else state['iteration']
    trace=[] if name=='parent6k' else state['sample_trace']
    before=deepcopy(old.cpu_values(old.snapshot(scene,opts,step,state['camera_stack'],trace)))
    return renderer,scene,opts,state,step,trace,before


def assert_no_update(scene,opts,state,step,trace,before):
    after=old.snapshot(scene,opts,step,state['camera_stack'],trace)
    old.assert_equivalent(before,after,0,0)
    assert all(p.grad is None for p in scene.parameters())


def native(renderer,scene,cam,t,colors=None,dynamic_only=False):
    dyn=scene.dynamic_at(t)
    bank=dyn if dynamic_only else compose(scene.static,[dyn])
    return renderer(cam,bank,PIPE,scene.centers.new_zeros(3),0.,0.,0.,override_color=colors)


def target_group(renderer,scene,cam,t):
    package=native(renderer,scene,cam,t,dynamic_only=True)
    rgb=package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
    alpha=package['alpha'].squeeze().cpu().numpy();mask=(alpha>.5)&(rgb.mean(2)<.05)
    ids=select_ids(package['gs_per_pixel'].cpu().numpy(),mask,len(scene.pools[0].get_xyz))
    colors=torch.from_numpy(packing_colors(len(scene.pools[0].get_xyz),0,[ids])).to('cuda')
    visible=native(renderer,scene,cam,t,colors=colors,dynamic_only=True)['render'][0].clamp_min(0).cpu().numpy()
    denominator=float(alpha[mask].astype(np.float64).sum());mass=float(visible[mask].astype(np.float64).sum())
    assert mass<=denominator+1e-3
    row={'pixels':int(mask.size),'dynamic_alpha_gt05_pixels':int((alpha>.5).sum()),
        'dark_high_alpha_pixels':int(mask.sum()),'selected_canonical_rows':len(ids),
        'target_dark_alpha_mass':denominator,'selected_native_visible_mass_on_dark_set':mass,
        'fraction_dark_alpha_explained_by_first20_ID_union':None if denominator==0 else mass/denominator,
        'remaining_mass_is_unrepresented_not_absent':True}
    return ids,row,(rgb,alpha,mask,visible)


def colors_for(scene,groups):
    return torch.from_numpy(packing_colors(len(scene.pools[0].get_xyz),len(scene.static.get_xyz),groups)).cuda()


def image_arrays(values):
    rgb,alpha,mask,visible=values
    arrays=[('native dynamic RGB',(rgb*255).round().astype('uint8'))]
    for name,array in [('dynamic alpha',alpha),('dark high-alpha set',mask.astype(float)),('selected group visibility',visible)]:
        arrays.append((name,np.repeat((np.clip(array,0,1)*255).round().astype('uint8')[...,None],3,2)))
    return arrays


def sheet(values,label,path):
    canvas=Image.new('RGB',(400*len(values),259),'#171923');draw=ImageDraw.Draw(canvas)
    draw.text((5,4),label,fill='white')
    for i,(name,array) in enumerate(values):
        draw.text((400*i+4,25),name,fill='white')
        canvas.paste(Image.fromarray(array).resize((400,225),Image.Resampling.LANCZOS),(400*i,34))
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def source_measure(renderer,scene,meta,index,groups):
    frame=meta['frames'][index];cam=camera(frame,meta,1600)
    observed=old.frame_target(frame).permute(1,2,0).cpu().numpy()
    rows=[];maps=[]
    for offset in [0,3]:
        colors=colors_for(scene,groups[offset:offset+3])
        package=native(renderer,scene,cam,frame['time'],colors)
        for channel in range(len(groups[offset:offset+3])):
            weight=package['render'][channel].clamp_min(0).cpu().numpy()
            rows.append(weighted_observation(weight,observed));maps.append(weight)
    return {'source_index':index,'file_path':frame['file_path'],'time':frame['time'],'groups':rows},maps,observed


@torch.no_grad()
def single_fixture(renderer):
    from ..nuisance_static.runtime import calibrated_camera
    p=read(BASE/'protocol.json')['native_single_control'];w,h=p['width'],p['height']
    cam=calibrated_camera(np.eye(4),np.array([[100.,0.,w/2],[0.,100.,h/2],[0.,0.,1.]]),w,h)
    bank=SimpleNamespace(get_xyz=torch.tensor([[0.,0.,p['z']]],device='cuda'),
        get_scaling=torch.full((1,3),p['scale'],device='cuda'),get_rotation=torch.tensor([[1.,0.,0.,0.]],device='cuda'),
        get_opacity=torch.full((1,1),p['opacity'],device='cuda'),get_features=torch.zeros((1,16,3),device='cuda'),
        max_sh_degree=3,active_sh_degree=3)
    package=renderer(cam,bank,PIPE,torch.zeros(3,device='cuda'),0.,0.,0.,override_color=torch.ones((1,3),device='cuda'))
    a=package['alpha'].squeeze();wgt=package['weight_per_gs_pixel'][0];ids=package['gs_per_pixel'][0]
    valid=ids==0;assert valid.any() and torch.equal(package['render'][0],a)
    def error(x):return float((wgt[valid]-x[valid]).abs().max())
    return {'single_native_white_RGB_equals_alpha_exact':True,'valid_pixels':int(valid.sum()),
        'alpha_max':float(a.max()),'stored_weight_vs_compositing_alpha_max_error':error(a),
        'stored_weight_vs_alpha_times_one_minus_alpha_max_error':error(a*(1-a)),
        'stored_weights_used_in_J5':False,'control_not_a_training_asset':True}


@torch.no_grad()
def qualify(replicate):
    verify();assert not stopped();torch.set_num_threads(4)
    renderer,scene,opts,state,step,trace,before=load('joint_depth_trust')
    meta=read(DATA/'transforms_train.json');byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
    groups=[];targets=[]
    for camid in [86,188]:
        for number in [2,100,200]:
            ids,row,_=target_group(renderer,scene,camera(byid[camid],meta,1600),(number-1)/199)
            groups.append(ids);targets.append(row)
    result,maps,_=source_measure(renderer,scene,meta,0,groups)
    # Same full-scene rendering path, and independently unpack each channel.
    frame=meta['frames'][0];cam=camera(frame,meta,1600)
    for i,group in enumerate(groups):
        unpack=native(renderer,scene,cam,frame['time'],colors_for(scene,[group]))['render'][0]
        assert torch.allclose(unpack,torch.from_numpy(maps[i]).cuda(),atol=1e-7,rtol=1e-6)
    a=scene.render(renderer,cam,frame['time']);b=native(renderer,scene,cam,frame['time'])
    assert all(torch.equal(a[k],b[k]) for k in ['render','alpha','depth'])
    fixture=single_fixture(renderer);assert_no_update(scene,opts,state,step,trace,before);verify()
    values={'source0':result,'targets':targets,'group_counts':[len(x) for x in groups],
        'group_hashes':[__import__('hashlib').sha256(x.tobytes()).hexdigest() for x in groups],
        'single_fixture':fixture,'packed_unpacked_native_visibility_passed':True,
        'same_native_RGB_alpha_D_exact':True,'model_twoAdam_RNG_sampler_graphs_no_update_exact':True}
    path=REPORT/f'qualification{replicate}.json';assert not path.exists();old.atomic_json(path,values)
    print('J5_QUALIFICATION_PASS',replicate,fixture,flush=True)


@torch.no_grad()
def audit():
    verify();assert not stopped();assert read(REPORT/'qualification.json')['passed']
    assert not (REPORT/'audit.json').exists();torch.set_num_threads(4)
    meta=read(DATA/'transforms_train.json');assert len(meta['frames'])==120
    census=metadata_census(meta['frames']);byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
    coverage=[]
    for number in [2,100,200]:
        t=(number-1)/199;times=np.array([f['time'] for f in meta['frames']]);delta=np.abs(times-t)
        coverage.append({'target_number':number,'time':t,'exact_train_images':int((delta==0).sum()),
            'nearest_train_time_gap':float(delta.min()),'nearest_training_files':[f['file_path'] for f,x in zip(meta['frames'],delta) if x==delta.min()]})
    result={'metadata':census,'fixed_time_coverage':coverage,'models':{},'images':{},'training_updates':0,
        'no_independent_geometry_or_physical_identity_GT':True,'source_sha256':sha(Path(__file__))}
    OUT.mkdir(parents=True,exist_ok=True)
    for name in read(BASE/'protocol.json')['models']:
        assert not stopped();renderer,scene,opts,state,step,trace,before=load(name)
        groups=[];targets=[]
        for camid in [86,188]:
            cam=camera(byid[camid],meta,1600)
            for number in [2,100,200]:
                assert not stopped();ids,row,values=target_group(renderer,scene,cam,(number-1)/199)
                groups.append(ids);row.update(camera=camid,number=number);targets.append(row)
                arrays=image_arrays(values);path=OUT/f'{name}_c{camid}_t{number}_diagnostic.png'
                sheet(arrays,f'J5 NO UPDATE {name} c{camid}/t{number}; diagnostic, not semantic label',path)
                result['images'][path.relative_to(ROOT).as_posix()]=sha(path)
                for label,array in arrays:
                    path=OUT/f'{name}_c{camid}_t{number}_{label.replace(" ","_")}_native.png'
                    Image.fromarray(array).save(path);result['images'][path.relative_to(ROOT).as_posix()]=sha(path)
        rows=[]
        for index in range(120):
            assert not stopped();row,maps,observed=source_measure(renderer,scene,meta,index,groups);rows.append(row)
            if index in [0,24,72]:
                values=[('observed training RGB',(observed*255).round().astype('uint8'))]
                values += [(f'group{i} visible contribution',np.repeat((np.clip(m,0,1)*255).round().astype('uint8')[...,None],3,2)) for i,m in enumerate(maps)]
                path=OUT/f'{name}_source{index:03d}_visibility.png';sheet(values,f'J5 NO UPDATE {name} source index{index}; groups overlap; not surfaces',path)
                result['images'][path.relative_to(ROOT).as_posix()]=sha(path)
            if index%20==0:print('J5_SOURCE',name,index,120,flush=True)
            old.atomic_json(RUNS/'state.json',{'status':'auditing','model':name,'source_completed':index+1,'pid':__import__('os').getpid()})
        totals=[]
        for group in range(6):
            mass=sum(x['groups'][group]['visible_weight'] for x in rows)
            def weighted(key):
                return None if mass==0 else sum(x['groups'][group]['visible_weight']*(x['groups'][group][key] or 0) for x in rows)/mass
            totals.append({'group':group,'source_count':120,'total_visible_weight':mass,
                'visible_nonzero_sources':sum(x['groups'][group]['visible_weight']>0 for x in rows),
                'weighted_observed_mean_RGB':weighted('observed_mean_RGB'),
                'weight_on_observed_RGB_lt005':weighted('weight_on_observed_RGB_lt005')})
        assert_no_update(scene,opts,state,step,trace,before)
        result['models'][name]={'targets':targets,'sources':rows,'source_totals':totals,
            'full_state_no_update_exact':True,'selected_row_IDs_not_exported':True}
        del scene,opts,before,renderer;__import__('gc').collect();torch.cuda.empty_cache()
    verify();old.atomic_json(REPORT/'audit.json',result)
    old.atomic_json(RUNS/'state.json',{'status':'audited_pending_independent_review','training_updates':0})
    print('J5_AUDIT_COMPLETE_PENDING_REVIEW',census['unique_times'],census['same_time_multicamera_times'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-rubber-visibility-audit',action='store_true')
    p.add_argument('--phase',choices=['qualify','audit'],default='qualify');p.add_argument('--replicate',type=int,choices=[0,1],default=0)
    args=p.parse_args()
    if not args.enable_rubber_visibility_audit:raise SystemExit('J5 default off; no data/model opened')
    # Reuse only the lock implementation with a task-local root.
    old.RUNS=RUNS
    with old.lock('worker.lock'):
        if args.phase=='qualify':qualify(args.replicate)
        else:audit()

if __name__=='__main__':main()
