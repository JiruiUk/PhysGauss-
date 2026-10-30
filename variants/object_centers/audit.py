"""Read-only C1 grouping candidates and train-frame center visualizations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck'
DATA=ROOT/'data/derived/kinematic_resplit/training'
OUTPUT=ROOT/'runs/pidg_object_centers/grouping_review/c1_final50000'
REPORTS=ROOT/'experiments/object_centers'
PALETTE=['#f26b8a','#26d6c5','#ffb347','#b794f6','#a7e36c','#50a4ff','#ed85e6','#e3d36c']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def tensor_image(value):
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.rint(value.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))


def main():
    options=argparse.ArgumentParser()
    options.add_argument('--only-motion',action='store_true',help='Append motion-dominant candidates to existing review')
    opts=options.parse_args()
    import numpy as np
    import torch
    from PIL import Image,ImageDraw,ImageFont,ImageColor
    from variants.object_centers.grouping import cluster_candidates
    from variants.object_centers.model import weighted_centers
    repo=ROOT/'variants/color_repair/code'
    sys.path.insert(0,str(repo));sys.path.insert(0,str(ROOT))
    os.environ['PIDG_METRIC_CONFIG']=str(SOURCE/'metric_config.json')
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    from variants.color_repair.run import assert_frozen
    assert_frozen(repo)
    from arguments import ModelParams,PipelineParams
    from scene_PIDG import Scene,DeformModel
    from gaussian_renderer import GaussianModel,render
    from variants.color_repair.color import colors as positive_colors
    from variants.coverage_tuning.data import frame_paths,read_alpha
    from variants.kinematic_tuning.model import center_trajectory
    np.random.seed(0);torch.manual_seed(0)
    OUTPUT.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    files=[SOURCE/'model_config.py',SOURCE/'metric_config.json',SOURCE/'point_cloud/iteration_50000/point_cloud.ply',
           SOURCE/'deform/iteration_50000/deform.pth']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    receipt=OUTPUT/'source.json'
    if receipt.exists() and json.loads(receipt.read_text())['source_sha256']!=hashes:
        raise ValueError('Existing audit belongs to different C1 weights')
    metadata=json.loads((DATA/'transforms_train.json').read_text())
    ordered=sorted(metadata['frames'],key=lambda f:int(Path(f['file_path']).name))
    frames=[ordered[i] for i in np.linspace(0,len(ordered)-1,12,dtype=int)]
    frame_ids=[int(Path(f['file_path']).name) for f in frames]
    loader=OUTPUT/'train_only_loader';loader.mkdir(exist_ok=True)
    selected=[]
    for frame in frames:
        f=dict(frame);f['file_path']=str((DATA/frame['file_path']).resolve());f['time']=(int(Path(frame['file_path']).name)-1)/199
        selected.append(f)
    write(loader/'transforms_train.json',{'camera_angle_x':metadata['camera_angle_x'],'frames':selected})
    write(loader/'transforms_test.json',{'camera_angle_x':metadata['camera_angle_x'],'frames':[]})
    shutil.copyfile(DATA/'points3d.ply',loader/'points3d.ply')
    write(receipt,{'source_sha256':hashes,'snapshot':50000,'frame_ids':frame_ids,
        'fit_split':'train','validation_RGB_loaded':False,'test_RGB_loaded':False,
        'meaning':'fixed unsupervised candidate groups, not true object IDs',
        'not_transferable_to_different_kernel_ordering':True})
    parser=argparse.ArgumentParser();model=ModelParams(parser);pipeline=PipelineParams(parser)
    args=parser.parse_args([])
    for key,value in runpy.run_path(str(SOURCE/'model_config.py')).items():
        if not key.startswith('__'):setattr(args,key,value)
    args.source_path=str(loader);args.model_path=str(SOURCE);args.resolution=800
    args.load2gpu_on_the_fly=True;args.data_device='cuda:0';args.eval=True
    paths=frame_paths(DATA,'train')
    with torch.no_grad():
        gaussians=GaussianModel(args.sh_degree)
        scene=Scene(model.extract(args),gaussians,load_iteration='50000',shuffle=False)
        assert len(scene.getTestCameras())==0
        views=sorted(scene.getTrainCameras(),key=lambda v:int(v.image_name))
        assert [int(v.image_name) for v in views]==frame_ids
        deform=DeformModel(args.grid_args,args.network_args,scale_xyz=args.scale_xyz,reg_spatial_able=False,reg_temporal_able=False)
        deform.load_weights(str(SOURCE),iteration='50000')
        canonical=gaussians.get_xyz.detach().cpu().numpy();opacity=gaussians.get_opacity.detach().flatten().cpu().numpy()
        poses=[];colors=[];votes=torch.zeros(len(canonical),device='cuda')
        background=torch.zeros(3,device='cuda');cached=[]
        for index,view in enumerate(views):
            view.load2device();view.fid=view.fid.new_tensor([(int(view.image_name)-1)/199])
            result=deform.step(gaussians.get_xyz.detach(),view.fid.unsqueeze(0).expand(len(canonical),-1),fixed_attention=True)
            package=render(view,gaussians,pipeline.extract(args),background,result['d_xyz'],result['d_rotation'],result['d_scaling'])
            target=read_alpha(paths[view.image_name],(view.image_width,view.image_height)).cuda()[0]
            support=(target>.95)&(package['alpha'][0]>.1)
            ids=package['gs_per_pixel'][:,support].reshape(-1).long()
            ids=ids[(ids>=0)&(ids<len(canonical))].unique();votes[ids]+=1
            posed=package['means3D_deform'];poses.append(posed.cpu().numpy())
            colors.append(positive_colors(gaussians,posed,view).clamp(0,1).cpu().numpy())
            cached.append((tensor_image(view.original_image[:3]),tensor_image(package['render'])))
            view.load2device('cpu')
            print(f'Extract train frame{view.image_name}: {index+1}/{len(views)}',flush=True)
        poses=np.stack(poses);colors=np.mean(colors,axis=0);votes=votes.cpu().numpy()
        eligible=(votes>=2)&(opacity>=.03)
        weights=np.clip(opacity*np.maximum(votes,1),1e-5,None).astype(np.float32)
        np.savez_compressed(OUTPUT/'kernel_evidence.npz',canonical=canonical,positions=poses,mean_colors=colors,
            foreground_candidate_votes=votes,opacity=opacity,eligible=eligible,weights=weights,frame_ids=np.array(frame_ids))
        candidates=[('canonical_K5',5,'canonical'),('trajectory_K5',5,'trajectory'),
                    ('trajectory_K6',6,'trajectory'),('trajectory_K8',8,'trajectory')]
        if opts.only_motion:
            candidates=[('motion_dominant_K5',5,'motion_dominant'),('motion_dominant_K6',6,'motion_dominant')]
        summaries=[]
        if opts.only_motion and (OUTPUT/'summary.json').exists():
            summaries=[s for s in json.loads((OUTPUT/'summary.json').read_text())['candidates'] if s['mode']!='motion_dominant']
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
        small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
        for name,k,mode in candidates:
            target=OUTPUT/name;target.mkdir(exist_ok=True)
            labels=cluster_candidates(canonical,poses,colors,eligible,weights,k,mode)
            np.save(target/'kernel_group_ids.npy',labels)
            ids_tensor=torch.from_numpy(labels.astype(np.int64)).cuda();w_tensor=torch.from_numpy(weights).cuda()
            centers=np.stack([weighted_centers(torch.from_numpy(posed).cuda(),ids_tensor,w_tensor,k).cpu().numpy() for posed in poses])
            np.savez_compressed(target/'center_trajectories.npz',frame_ids=frame_ids,times=(np.array(frame_ids)-1)/199,
                                centers=centers,group_ids=np.arange(k))
            palette=np.array([ImageColor.getrgb(c) for c in PALETTE[:k]],dtype=np.float32)/255
            rgb=np.tile(np.array([.35,.4,.46],dtype=np.float32),(len(labels),1))
            rgb[eligible]=palette[labels[eligible]]
            override=torch.from_numpy(rgb).cuda()
            groups=[]
            for group in range(k):
                mask=labels==group;point_colors=colors[mask]
                rms=float(np.sqrt(np.mean(np.sum((poses[:,mask]-poses[0,mask][None])**2,axis=-1))))
                center_rms=float(np.sqrt(np.mean(np.sum((centers[:,group]-centers[0,group])**2,axis=-1))))
                groups.append({'id':group,'color':PALETTE[group],'kernels':int(mask.sum()),
                    'mean_rgb':point_colors.mean(0).tolist(),'kernel_motion_rms':rms,'center_motion_rms':center_rms,
                    'yellow_like_fraction':float(((point_colors[:,0]>point_colors[:,2]+.1)&(point_colors[:,1]>point_colors[:,2]+.1)).mean()),
                    'blue_like_fraction':float((point_colors[:,2]>point_colors[:,0]+.1).mean())})
            for index,view in enumerate(views):
                view.load2device()
                result=deform.step(gaussians.get_xyz.detach(),view.fid.unsqueeze(0).expand(len(canonical),-1),fixed_attention=True)
                package=render(view,gaussians,pipeline.extract(args),background,result['d_xyz'],result['d_rotation'],result['d_scaling'],override_color=override)
                colored=tensor_image(package['render'])
                draw=ImageDraw.Draw(colored)
                c=torch.from_numpy(centers[index]).cuda();h=torch.cat((c,torch.ones(k,1,device='cuda')),-1)@view.full_proj_transform
                ndc=h[:,:2]/h[:,3:4];pixels=((ndc+1)*c.new_tensor([view.image_width,view.image_height])/2-.5).cpu().numpy()
                for group,(x,y) in enumerate(pixels):
                    if h[group,3]>0 and 0<=x<view.image_width and 0<=y<view.image_height:
                        x,y=int(x),int(y);draw.ellipse((x-6,y-6,x+6,y+6),fill=PALETTE[group],outline='white',width=2)
                        draw.text((x+8,y-12),f'G{group}',font=small,fill='white',stroke_width=2,stroke_fill='black')
                width,height=colored.size
                canvas=Image.new('RGB',(width*3,height+95),'#edf2f7');d=ImageDraw.Draw(canvas)
                for col,(text,image) in enumerate(zip(['GT / train frame'+view.image_name,'C1 final50000',name+' / fixed membership'],[*cached[index],colored])):
                    d.text((col*width+15,12),text,font=font,fill='#17304a');canvas.paste(image,(col*width,50))
                for group in range(k):
                    x=15+group*190;d.rectangle((x,height+62,x+18,height+80),fill=PALETTE[group])
                    d.text((x+26,height+60),f'G{group}',font=small,fill='#17304a')
                d.text((k*190+35,height+60),'Gray: unassigned',font=small,fill='#52677a')
                canvas.save(target/f'{index:05d}.png',compress_level=3)
                view.load2device('cpu')
            summary={'name':name,'groups':groups,'K':k,'mode':mode,'assigned_kernels':int(eligible.sum()),
                'unassigned_kernels':int((labels<0).sum()),'total_kernels':len(labels),
                'membership_fixed_across_time':True,'true_instance_labels':False}
            write(target/'summary.json',summary);summaries.append(summary)
            plot_centers(target,centers,frame_ids,k)
            print('Rendered candidate '+name,flush=True)
    for p in files:
        if digest(p)!=hashes[str(p.relative_to(ROOT))]:raise ValueError('Original C1 source changed')
    assert_frozen(repo)
    summary={'status':'completed','source':'C1_repaired final50000','training_frame_ids':frame_ids,
        'validation_RGB_used':False,'test_RGB_used':False,'original_C1_weights_unchanged':True,
        'fixed_candidate_grouping_not_semantic_ground_truth':True,'new_center_training_started':False,
        'foreground_support_rule':'At least2/12 training-frame foreground contributor-candidate votes; opacity>=0.03',
        'center_rule':'fixed opacity*vote weighted mean of assigned kernel centers; not physical center of mass',
        'feature_rule':'canonical XYZ OR reference posed XYZ + scaled12-time displacements + mean SH RGB; motion_dominant uses XYZ*0.15, displacement*4, RGB*0.75',
        'candidates':summaries,'source_sha256':hashes,'output_absolute':str(OUTPUT),
        'gallery_absolute':str(OUTPUT/'index.html')}
    write(OUTPUT/'summary.json',summary);write(REPORTS/'grouping_review.json',summary)
    make_gallery(summary)
    review_artifacts(summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('candidates','source_sha256')},indent=2),flush=True)


def plot_centers(target,centers,frame_ids,k):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
    for group in range(k):
        c=centers[:,group]
        for ax,(a,b) in zip(axes[:2],[(0,1),(0,2)]):
            ax.plot(c[:,a],c[:,b],'-o',ms=3,color=PALETTE[group],label=f'G{group}')
        axes[2].plot(frame_ids,np.linalg.norm(c-c[0],axis=-1),'-o',ms=3,color=PALETTE[group],label=f'G{group}')
    axes[0].set(xlabel='world x',ylabel='world y',title='Center trajectory XY')
    axes[1].set(xlabel='world x',ylabel='world z',title='Center trajectory XZ')
    axes[2].set(xlabel='Original training frame',ylabel='Center displacement',title='Trend from current C1 (not new training)')
    for ax in axes:ax.grid(alpha=.2)
    axes[2].legend(ncol=2,fontsize=8);fig.savefig(target/'center_trends.png',dpi=160);plt.close(fig)


def make_gallery(summary):
    names=[c['name'] for c in summary['candidates']]
    data={c['name']:c for c in summary['candidates']}
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C1 object-center grouping review</title><style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:24px}img{width:100%;height:auto}a{color:#8de4b1}select,button{padding:8px;margin:8px}input{width:35%}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:8px;border-bottom:1px solid #52677a}</style>
<h1>C1: fixed object-center candidates</h1><p>Candidate groups are NOT ground-truth object IDs. GT / C1 render / fixed group colors + projected3D centers.</p>
<p>Only12 training frames used. Gray kernels are unassigned. Center curves summarize existing C1 predictions; no new center/collision model has been trained.</p>
<p style="padding:16px;background:#593b26">检查结论：可见鸭子合组、同一物体被拆组，尚不适合作为组间排斥标签。六种候选供检查；未开始新的长训练。</p>
<p><a href="overview_frame0002.png">六候选：frame2</a> | <a href="overview_frame0146.png">frame146</a> | <a href="overview_frame0200.png">frame200</a> | <a href="qa.json">Verification receipt</a></p>
<select id="candidate">OPTIONS</select><button id="prev">Previous</button><input id="frame" type="range" min="0" max="11" value="0"><button id="next">Next</button><span id="number"></span> <a id="full">Open PNG</a>
<img id="image"><h2>Group statistics</h2><p id="support"></p><table><thead><tr><th>Group</th><th>Kernels</th><th>Yellow-like fraction</th><th>Blue-like fraction</th><th>Center motion RMS</th></tr></thead><tbody id="rows"></tbody></table>
<h2>Candidate center trends</h2><img id="trends"><p><a id="labels">Kernel ID file (.npy)</a> | <a href="summary.json">Full audit</a></p>
<script>const data=DATA,ids=IDS,c=document.getElementById('candidate'),f=document.getElementById('frame');function update(){const n=String(f.value).padStart(5,'0'),item=data[c.value],p=c.value+'/'+n+'.png';document.getElementById('image').src=p;document.getElementById('full').href=p;document.getElementById('trends').src=c.value+'/center_trends.png';document.getElementById('labels').href=c.value+'/kernel_group_ids.npy';document.getElementById('number').textContent='Original training frame '+ids[Number(f.value)]+' | '+(Number(f.value)+1)+'/12';document.getElementById('support').textContent=item.assigned_kernels+'/'+item.total_kernels+' kernels assigned; '+item.unassigned_kernels+' remain unassigned.';document.getElementById('rows').innerHTML=item.groups.map(g=>'<tr><td style="color:'+g.color+'">G'+g.id+'</td><td>'+g.kernels+'</td><td>'+g.yellow_like_fraction.toFixed(3)+'</td><td>'+g.blue_like_fraction.toFixed(3)+'</td><td>'+g.center_motion_rms.toFixed(4)+'</td></tr>').join('');}function step(n){f.value=Math.max(0,Math.min(11,Number(f.value)+n));update();}c.onchange=f.oninput=update;document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);update();</script></html>'''
    page=page.replace('OPTIONS',''.join('<option>'+n+'</option>' for n in names)).replace('DATA',json.dumps(data)).replace('IDS',json.dumps(summary['training_frame_ids']))
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')


def review_artifacts(summary):
    """Check exported panels and source hashes, and create compact comparisons."""
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    candidates=summary['candidates']
    for relative,expected in summary['source_sha256'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Frozen source changed: '+relative)
    panel_count=0
    for candidate in candidates:
        folder=OUTPUT/candidate['name']
        labels=np.load(folder/'kernel_group_ids.npy')
        if labels.shape!=(candidate['total_kernels'],):raise ValueError('Misaligned labels')
        if int((labels>=0).sum())!=candidate['assigned_kernels']:raise ValueError('Assignment count changed')
        curves=np.load(folder/'center_trajectories.npz')
        if curves['centers'].shape!=(12,candidate['K'],3) or not np.isfinite(curves['centers']).all():
            raise ValueError('Invalid center trajectories')
        for index in range(12):
            path=folder/f'{index:05d}.png'
            with Image.open(path) as im:
                if im.size!=(2400,545):raise ValueError('Unexpected triptych dimensions')
                with Image.open(OUTPUT/candidates[0]['name']/path.name) as reference:
                    if not np.array_equal(np.asarray(im)[50:500,:1600],np.asarray(reference)[50:500,:1600]):
                        raise ValueError('GT/C1 panels differ across candidates')
            panel_count+=1
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for index in (0,8,11):
        canvas=Image.new('RGB',(1920,390*((len(candidates)+2)//3)),'#edf2f7')
        draw=ImageDraw.Draw(canvas)
        for j,c in enumerate(candidates):
            x,y=(j%3)*640,(j//3)*390
            with Image.open(OUTPUT/c['name']/f'{index:05d}.png') as im:
                panel=im.crop((1600,50,2400,500)).resize((640,360),Image.Resampling.LANCZOS)
            draw.text((x+8,y+3),c['name'],font=font,fill='#17304a');canvas.paste(panel,(x,y+30))
        canvas.save(OUTPUT/f"overview_frame{summary['training_frame_ids'][index]:04d}.png")
    code_files=list((ROOT/'variants/object_centers').glob('*.py'))+[ROOT/'variants/object_centers/README.md',ROOT/'tests/test_object_centers.py']
    receipt={'triptych_pngs_checked':panel_count,'GT_and_C1_panels_identical_across_candidates':True,
             'original_model_hashes_match':True,'validation_or_test_RGB_loaded':False,
             'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in code_files},
             'visual_review':'Merged ducks, split ducks, and split blue geometry; no candidate accepted as physical object IDs',
             'training_integration':'none','long_training_started':False}
    write(OUTPUT/'qa.json',receipt)


if __name__=='__main__':main()
