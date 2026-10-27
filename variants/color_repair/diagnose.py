"""Read-only original checkpoints: raw SH and float RGB on training frames."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);opts=parser.parse_args()
    repo=ROOT/'variants/coverage_tuning/code'
    sys.path.insert(0,str(repo));sys.path.insert(0,str(ROOT))
    os.environ['PIDG_HASH_SOURCE_DIR']=str(ROOT/'third_party/PIDG/hashencoder')
    os.environ['PIDG_HASH_BUILD_DIR']=str(ROOT/'third_party/PIDG/tmp_build')
    import numpy as np
    import torch
    from arguments import ModelParams,PipelineParams
    from scene_PIDG import Scene,DeformModel
    from gaussian_renderer import GaussianModel,render
    from variants.color_repair.color import raw_sh_color
    from variants.coverage_tuning.data import frame_paths,read_alpha
    from variants.kinematic_resplit.runtime import validate_scene
    names=['C1_mask','C2_mask_affine','C3_mask_affine_density'];result={}
    source=ROOT/'data/derived/kinematic_resplit/training'
    for name in names:
        directory=ROOT/'runs/pidg_coverage_tuning'/name/'rubber_duck'
        weights=[directory/'point_cloud/iteration_50000/point_cloud.ply',directory/'deform/iteration_50000/deform.pth']
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in weights}
        argparser=argparse.ArgumentParser();model=ModelParams(argparser);pipeline=PipelineParams(argparser)
        args=argparser.parse_args([])
        for key,value in runpy.run_path(str(directory/'model_config.py')).items():
            if not key.startswith('__'):setattr(args,key,value)
        args.source_path=str(source);args.model_path=str(directory);args.resolution=400
        args.load2gpu_on_the_fly=True;args.data_device='cuda:0';args.eval=True
        with torch.no_grad():
            gaussian=GaussianModel(args.sh_degree)
            scene=Scene(model.extract(args),gaussian,load_iteration='50000',shuffle=False)
            validate_scene(scene,source)
            deform=DeformModel(args.grid_args,args.network_args,scale_xyz=args.scale_xyz,reg_spatial_able=False,reg_temporal_able=False)
            deform.load_weights(str(directory),iteration='50000')
            paths=frame_paths(source,'train');views=sorted(scene.getTrainCameras(),key=lambda v:v.image_name)
            rows=[]
            for index in np.linspace(0,len(views)-1,9,dtype=int):
                view=views[index];view.load2device()
                t=view.fid.unsqueeze(0).expand(len(gaussian.get_xyz),-1)
                warp=deform.step(gaussian.get_xyz.detach(),t,fixed_attention=True)
                package=render(view,gaussian,pipeline.extract(args),torch.zeros(3,device='cuda'),
                    warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])
                raw=raw_sh_color(gaussian,package['means3D_deform'],view)
                mask=read_alpha(paths[view.image_name],(view.image_width,view.image_height)).to('cuda')[0]>.95
                gt=view.original_image[:3].to('cuda')
                red_target=mask&(gt[0]>.25)
                ids=package['gs_per_pixel'][:,mask].reshape(-1).long()
                ids=ids[(ids>=0)&(ids<len(raw))].unique()
                rows.append({'training_frame':view.image_name,'foreground_contributor_count':len(ids),
                    'foreground_contributors_negative_raw_rgb_fraction':(raw[ids]<0).float().mean(0).tolist(),
                    'minimum_raw_rgb':raw[ids].min(0).values.tolist(),
                    'zero_float_red_where_gt_red_positive':float(((package['render'][0]<=1e-8)&red_target).sum().item()/max(1,red_target.sum().item())),
                    'sample_negative_red_contributors':[{'id':int(i),'raw_rgb':raw[i].tolist()} for i in ids[raw[ids,0]<0][:8].tolist()]})
                view.load2device('cpu')
            result[name]={'snapshot':'50000','split':'train','frame_count':9,'per_frame':rows,
                'mean_negative_rgb_contributor_fraction':np.mean([r['foreground_contributors_negative_raw_rgb_fraction'] for r in rows],axis=0).tolist(),
                'mean_zero_red_fraction_where_gt_red_positive':float(np.mean([r['zero_float_red_where_gt_red_positive'] for r in rows])),
                'source_weights_sha256':hashes}
            for p in weights:
                if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)]:raise ValueError('Diagnostic mutated original weights')
            del scene,gaussian,deform,package,raw
            torch.cuda.empty_cache()
    opts.output.parent.mkdir(parents=True,exist_ok=True)
    opts.output.write_text(json.dumps({'status':'completed','resolution_width':400,
        'scope':'Nine predetermined training frames per model; contributor IDs are visibility candidates, not exact per-pixel causal attribution.',
        'models':result},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({n:{k:v for k,v in r.items() if k.startswith('mean_')} for n,r in result.items()},indent=2))


if __name__=='__main__':main()
