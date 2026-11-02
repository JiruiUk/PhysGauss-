"""Full train-sequence audit of frozen models; never starts training."""
from pathlib import Path
import os,sys,json,runpy,gc,argparse,hashlib
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_support.core import components,propagate,RenderProxy,ratio
OUT=ROOT/'output/surface_support';REPORT=ROOT/'experiments/surface_support'
TRAIN=ROOT/'data/derived/kinematic_resplit/training';CACHE=ROOT/'data/derived/flow_comparison_train_only'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
MODELS={
 'C1_final':(ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck',ROOT/'output/shape_coherence/audit/reference_graph.pt','b5d4cf5c9fb8fba4fcc26124555d48aa7d3a963a8be1d28488954752db486480'),
 'P1_16k':(ROOT/'runs/pidg_shape_coherence/P1_local_shape/rubber_duck',ROOT/'runs/pidg_shape_coherence/P1_local_shape/rubber_duck/shape_reference/reference_graph.pt','b36bfde8e25ffe1057e493a4bcbb40c17236882151e07118f95d091048a82b17')}
PALETTE=np.array([[235,90,100],[40,210,180],[245,170,65],[175,130,245]])/255
SELECT={2,19,50,62,74,86,99,108,146,200}


def rgba(frame,width=400):
    with Image.open(TRAIN/(frame['file_path']+'.png')) as im:return np.asarray(im.convert('RGBA').resize((width,width*9//16),Image.Resampling.BILINEAR))


def inputs(meta,manifest):
    files=[TRAIN/'transforms_train.json',CACHE/'manifest.json',REPORT/'plan.json',Path(__file__),OUT/'core.py',ROOT/'tests/test_surface_support.py']
    files += [TRAIN/(f['file_path']+'.png') for f in meta['frames']]
    for pair in manifest['pairs']:
        p=CACHE/pair['file'];assert sha(p)==manifest['output_sha256'][pair['file']];files.append(p)
    for name,(directory,graph,expected) in MODELS.items():
        p=directory/'checkpoints/resume.pt';assert sha(p)==expected
        files += [p,p.with_suffix('.json'),graph,directory/'model_config.py',directory/'metric_config.json']
    for rel,value in manifest['input_sha256'].items():
        if rel.endswith('.png') or rel.endswith('transforms_train.json'):assert sha(ROOT/rel)==value
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def track_images(meta,manifest):
    """Image evidence only; cached depth-compensated fields are never read."""
    frame_ids=[int(Path(f['file_path']).stem) for f in meta['frames']]
    comps=[components(rgba(f)) for f in meta['frames']]
    assert len(frame_ids)==120 and frame_ids==manifest['train_frame_ids']
    labels=comps[0].copy();assert labels.max()==3
    tracks=[labels.copy()];rows=[]
    rows.append({'frame':frame_ids[0],'components':4,'tracked_pixels':[int((labels==k).sum()) for k in range(4)],'ambiguous_components':[]})
    for i,pair in enumerate(manifest['pairs']):
        assert [pair['source'],pair['target']]==frame_ids[i:i+2]
        with np.load(CACHE/pair['file']) as data:
            forward=torch.from_numpy(data['forward'].copy());backward=torch.from_numpy(data['backward'].copy())
        labels,amb=propagate(labels,comps[i+1],forward,backward)
        tracks.append(labels.copy());rows.append({'frame':frame_ids[i+1],'components':int(comps[i+1].max()+1),'tracked_pixels':[int((labels==k).sum()) for k in range(4)],'ambiguous_components':amb})
    np.savez_compressed(OUT/'image_tracks.npz',frames=frame_ids,labels=np.stack(tracks).astype(np.int8),components=np.stack(comps).astype(np.int8))
    report={'method':'Visible-only estimated raw FB flow; nearest pixel sampling is only image advection, never nearest-kernel ID assignment','no_reidentification_after_occlusion':True,'cached_motion_and_depth_valid_not_used':True,'frames':rows}
    write(REPORT/'image_tracking.json',report)
    return comps,tracks


def panel(values,titles,destination):
    w,h=values[0].shape[1],values[0].shape[0]
    image=Image.new('RGB',(w*len(values),h+40),'#182a3d');draw=ImageDraw.Draw(image)
    for i,(a,title) in enumerate(zip(values,titles)):
        image.paste(Image.fromarray((np.clip(a,0,1)*255).round().astype('uint8')),(i*w,40));draw.text((i*w+6,8),title,fill='white')
    image.save(destination)


@torch.no_grad()
def audit_model(name,meta,comps,tracks,probe=False):
    from variants.c1_flow_ablation.results import motion_module
    motion_module()
    from scene_PIDG.gaussian_model import GaussianModel
    from scene_PIDG.deform_model import DeformModel
    from reproduction.scripts.training_checkpoint import load_state,restore_models
    from variants.shape_coherence.observations import camera,PIPE
    from gaussian_renderer import render
    from variants.flow_comparison.temporal import warp_at
    from variants.kinematic_tuning.model import center_trajectory
    from variants.flow_comparison.flow import projected,rotation_raw
    from variants.unified_flow.flow import contribution_weights
    directory,graphpath,_=MODELS[name]
    state=load_state(directory/'checkpoints/resume.pt');args=runpy.run_path(str(directory/'model_config.py'))
    g=GaussianModel(3);d=DeformModel(args['grid_args'],args['network_args'],reg_spatial_able=False,reg_temporal_able=False)
    opt=SimpleNamespace(**state['opt']);d.train_setting(opt);restore_models(state,g,d,opt)
    iteration=state['iteration'];del state;gc.collect()
    os.environ['PIDG_METRIC_CONFIG']=str(directory/'metric_config.json')
    graph=torch.load(graphpath,map_location='cuda',weights_only=True);labels=graph['labels'];known=labels>=0;ids=torch.where(known)[0]
    bg=torch.zeros(3,device='cuda');white=torch.ones(len(labels),3,device='cuda')
    group_colors=torch.zeros_like(white);fourth=torch.zeros_like(white)
    for k in range(3):group_colors[:,k]=(labels==k)
    fourth[:,0]=(labels==3)
    isolated=RenderProxy(g,get_opacity=g.get_opacity*known[:,None])
    refwarp=warp_at(d,g.get_xyz.detach(),1/199);refscale=(g.get_scaling+refwarp['d_scaling']).clone();refrot=(g.get_rotation+refwarp['d_rotation']).clone()
    fixed=RenderProxy(isolated,get_scaling=refscale,get_rotation=refrot)
    refcam=camera(meta['frames'][0],meta,400)
    # Late visible probes are selected by actual rendered contribution, not
    # spatial nearest neighbor. Their component IDs are LOCAL TO FRAME86 ONLY.
    lateframe=next(f for f in meta['frames'] if int(Path(f['file_path']).stem)==86)
    latecam=camera(lateframe,meta,400);latewarp=warp_at(d,g.get_xyz.detach(),85/199)
    latepkg=render(latecam,g,PIPE,bg,latewarp['d_xyz'],latewarp['d_rotation'],latewarp['d_scaling'])
    si,we=contribution_weights(latepkg,g.get_opacity);ci=next(i for i,f in enumerate(meta['frames']) if f==lateframe)
    late_ids=[];late_components=[]
    for k in range(int(comps[ci].max()+1)):
        mass=torch.zeros(len(labels),device='cuda');mk=torch.from_numpy(comps[ci]==k).cuda()
        mass.scatter_add_(0,si.reshape(-1),(we*mk[None]).reshape(-1));mass[known]=0
        chosen=torch.argsort(mass,descending=True)[:20];chosen=chosen[mass[chosen]>.02]
        for index in chosen.tolist():
            if index not in late_ids:late_ids.append(index);late_components.append(k)
    late_ids=torch.tensor(late_ids,device='cuda',dtype=torch.long)
    late_color=torch.zeros_like(white);late_color[late_ids]=1
    del latepkg,si,we,mass
    saved=[];trajectories=[];pixel_centers=[];lower_visibility=[];footprints=[];late_centers=[];rows=[];previous=None
    destination=OUT/name;destination.mkdir(parents=True,exist_ok=True)
    frame_list=meta['frames'] if not probe else [f for f in meta['frames'] if int(Path(f['file_path']).stem) in {2,74,86,200}]
    ref_eig=None
    for frame in frame_list:
        number=int(Path(frame['file_path']).stem);i=next(i for i,f in enumerate(meta['frames']) if f==frame)
        cam=camera(frame,meta,400);warp=warp_at(d,g.get_xyz.detach(),(number-1)/199);centers=center_trajectory(g.get_xyz.detach(),warp['d_xyz'])
        def rr(model=g,color=None,view=cam,footprint=False):
            return render(view,model,PIPE,bg,warp['d_xyz'],torch.zeros_like(warp['d_rotation']) if footprint else warp['d_rotation'],torch.zeros_like(warp['d_scaling']) if footprint else warp['d_scaling'],override_color=color)
        rgb=rr();safe,weights=contribution_weights(rgb,g.get_opacity)
        total=rgb['alpha'].squeeze();mask=torch.from_numpy(comps[i]>=0).cuda();den=total[mask].sum()
        a=rr(color=group_colors)['render'];b=rr(color=fourth)['render'][0]
        exact=a.sum(0)+b
        late=rr(color=late_color)['render'][0]
        assert float((exact-total).max())<2e-4
        low=(weights*known[safe]).sum(0);represented=weights.sum(0)
        iso=rr(isolated,white)['render'][0];fix=rr(fixed,white,footprint=True)['render'][0]
        viewfixed=rr(isolated,white,view=refcam)['render'][0]
        partial=torch.zeros(len(labels),device='cuda');partial.scatter_add_(0,safe.reshape(-1),(weights*mask[None]).reshape(-1))
        active=partial>.02
        mean,cov=projected(centers[known],(g.get_scaling+warp['d_scaling'])[known],(g.get_rotation+warp['d_rotation'])[known],cam)
        view=cam.world_view_transform;depth=(centers[known]@view[:3,:3]+view[3,:3])[:,2]
        inside=(depth>.2)&(mean[:,0]>=0)&(mean[:,0]<400)&(mean[:,1]>=0)&(mean[:,1]<225)
        xy=mean.round().long();onmask=inside&mask[xy[:,1].clamp(0,224),xy[:,0].clamp(0,399)]
        rot=rotation_raw((g.get_rotation+warp['d_rotation'])[known]);sc=(g.get_scaling+warp['d_scaling'])[known]
        eig=torch.linalg.eigvalsh(((rot*sc[:,None,:].square())@rot.transpose(-1,-2)).double()).clamp_min(1e-20)
        if ref_eig is None:ref_eig=eig.clone()
        area=torch.linalg.det(cov).clamp_min(0).sqrt();r=rgba(frame);gt=torch.from_numpy(r[...,:3].astype(np.float32)/255*(r[...,3:4]/255)).cuda()
        row={'frame':number,'duck_pixels':int(mask.sum()),'exact_known_share':ratio(exact[mask].sum(),den),
             'first20_known_share_true_alpha':ratio(low[mask].sum(),den),'old_first20_normalized_known_share':ratio(low[mask].sum(),represented[mask].sum()),
             'first20_total_share':ratio(represented[mask].sum(),den),'exact_group_share':[ratio(x[mask].sum(),den) for x in [*a,b]],
             'late86_local_probe_exact_share':ratio(late[mask].sum(),den),
             'early_group_mass_on_visible_image_tracks':[[float(x[torch.from_numpy(tracks[i]==k).cuda()].sum()) for k in range(4)] for x in [*a,b]],
             'isolated_known_duck_mass':float(iso[mask].sum()),'actual_known_duck_mass':float(exact[mask].sum()),
             'fixed_clear_footprint_duck_mass':float(fix[mask].sum()),'isolated_known_all_pixels_mass':float(iso.sum()),'fixed_camera_isolated_all_pixels_mass':float(viewfixed.sum()),
             'external_occlusion_retention':ratio(exact[mask].sum(),iso[mask].sum()),
             'known_centers_in_frustum_fraction':float(inside.float().mean()),'known_centers_on_duck_fraction':float(onmask.float().mean()),
             'world_cov_linear_scale_ratio_p05_p50_p95':torch.quantile((eig/ref_eig).sqrt().flatten(),torch.tensor([.05,.5,.95],device='cuda',dtype=torch.float64)).tolist(),
             'projected_footprint_sqrt_det_p50_p95':torch.quantile(area,torch.tensor([.5,.95],device='cuda')).tolist(),
             'visible_duck_kernel_count_first20_lower_bound':int(active.sum()),'known_visible_kernel_count_first20_lower_bound':int((active&known).sum()),
             'adjacent_visible_kernel_jaccard_lower_bound':None if previous is None else ratio((active&previous).sum(),(active|previous).sum()),
             'duck_rgb_mae':float((rgb['render'].permute(1,2,0)[mask]-gt[mask]).abs().mean())}
        previous=active;rows.append(row);saved.append(number);trajectories.append(centers[known].cpu().numpy());pixel_centers.append(mean.cpu().numpy());lower_visibility.append(partial[known].cpu().numpy());footprints.append(area.cpu().numpy());late_centers.append(centers[late_ids].cpu().numpy())
        group_rgb=a.permute(1,2,0).cpu().numpy()@PALETTE[:3]+b.cpu().numpy()[...,None]*PALETTE[3]
        track=np.zeros((225,400,3))
        for k in range(4):track[tracks[i]==k]=PALETTE[k]
        # Isolated/fixed alpha channels are counterfactuals, not object GT masks.
        panel([gt.cpu().numpy(),rgb['render'].permute(1,2,0).clamp(0,1).cpu().numpy(),group_rgb,np.stack([exact.cpu().numpy(),iso.cpu().numpy(),fix.cpu().numpy()],-1),track],
              [f'{name} train GT {number}',f'render {iteration}',f'early patch full contribution', 'R actual G isolated B fixed size','raw FB visible ID evidence'],destination/f'frame{number:04d}.png')
        if number in SELECT:
            np.savez_compressed(destination/f'maps{number:04d}.npz',actual=exact.cpu().numpy(),isolated=iso.cpu().numpy(),fixed_footprint=fix.cpu().numpy(),fixed_camera=viewfixed.cpu().numpy(),true_alpha=total.cpu().numpy(),first20=represented.cpu().numpy())
        if number in SELECT or len(rows)%10==0:print(name,number,json.dumps({k:row[k] for k in ['exact_known_share','first20_total_share','external_occlusion_retention','known_centers_on_duck_fraction']}),flush=True)
        del rgb,safe,weights,a,b,iso,fix,viewfixed,partial,late
    np.savez_compressed(destination/'trajectories.npz',frames=saved,kernel_ids=ids.cpu().numpy(),labels=labels[known].cpu().numpy(),centers=trajectories,projected_centers=pixel_centers,first20_visibility_lower_bound=lower_visibility,projected_footprint=footprints,late86_probe_ids=late_ids.cpu().numpy(),late86_components_local_only=late_components,late86_probe_centers=late_centers)
    resolution=[]
    for frame in [f for f in meta['frames'] if int(Path(f['file_path']).stem) in {2,74,86,200}]:
        number=int(Path(frame['file_path']).stem);cam=camera(frame,meta,800);warp=warp_at(d,g.get_xyz.detach(),(number-1)/199)
        color=known[:,None].float().expand(-1,3).contiguous()
        p=render(cam,g,PIPE,bg,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=color)
        r=rgba(frame,800);cm=components(r);mk=torch.from_numpy(cm>=0).cuda();den=p['alpha'].squeeze()[mk].sum()
        exact800=float(p['render'][0][mk].sum());alpha800=float(den)
        ss,ww=contribution_weights(p,g.get_opacity)
        resolution.append({'frame':number,'width':800,'exact_known_share':ratio(exact800,alpha800),'first20_total_share':ratio(ww.sum(0)[mk].sum(),den)})
        del p,ss,ww
    report={'model':name,'iteration':iteration,'checkpoint_sha256':sha(directory/'checkpoints/resume.pt'),'graph_sha256':sha(graphpath),
            'point_count':len(labels),'early_partial_patch_counts':[int((labels==k).sum()) for k in range(4)],'resolution':[400,225],
            'late_probe_rule':'Top20 positive first20 contribution kernels per image component at frame86, excluding early known kernels; local visible probes, never persistent object IDs',
            'late_probe_count':len(late_ids),'resolution_checks':resolution,
            'note':'Full compositor cohort mass is exact at this resolution. Per-kernel visibility is first20 LOWER BOUND. Counterfactuals do not identify true physical occlusion or material identity.', 'frames':rows}
    write(REPORT/(name+('_probe' if probe else '')+'.json'),report)
    del g,d,graph;gc.collect();torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    from variants.shape_coherence.run import assert_frozen
    assert_frozen(ROOT/'variants/shape_coherence/code')
    from output.c1_flow_ablation.reuse_control import control_hashes
    control=control_hashes();assert control==read(ROOT/'experiments/c1_flow_ablation/reuse_control_plan.json')['historical_control_sha256']
    meta=read(TRAIN/'transforms_train.json');manifest=read(CACHE/'manifest.json')
    before=inputs(meta,manifest);write(REPORT/'input_lock.json',before)
    comps,tracks=track_images(meta,manifest)
    results=[audit_model(name,meta,comps,tracks,args.probe) for name in MODELS]
    assert inputs(meta,manifest)==before and control_hashes()==control
    write(REPORT/('probe_verification.json' if args.probe else 'verification.json'),{'passed':True,'models_unchanged':True,'protected_historical_control_unchanged':True,'frozen_source_and_data_passed':True,'all_training_frames':not args.probe,'frames_each':[len(r['frames']) for r in results],'read_only_no_optimizer_update':True,'input_hashes':before,'manual_review_pending':True})
    print('READ-ONLY AUDIT COMPLETE',flush=True)


if __name__=='__main__':main()
