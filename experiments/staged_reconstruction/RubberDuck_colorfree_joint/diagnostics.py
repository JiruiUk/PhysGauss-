"""Independent native diagnostics adapted from the frozen J6 evaluator."""
from pathlib import Path
import torch
from PIL import Image
from variants.staged_reconstruction.colorfree_joint import runtime as a
from variants.staged_reconstruction.prepare_data import ROOT,DATA,read,sha
from variants.staged_reconstruction.train_static import camera,render_static,PIPE
from experiments.staged_reconstruction.RubberDuck_joint_pilot.evaluate import rgb,alpha,depth,sheet
r=a.r

@torch.no_grad()
def diagnostics(renderer,models):
    meta=read(DATA/'transforms_train.json');byid={int(Path(f['file_path']).stem):f for f in meta['frames']}
    images={};camera_receipts={};support=[]
    for camid in [86,188]:
        frame=byid[camid];cam=camera(frame,meta,1600)
        camera_receipts[str(camid)]={'frame':frame['file_path'],'camera_transform':frame['transform_matrix'],
            'FoVx':cam.FoVx,'FoVy':cam.FoVy,'size':[1600,900]}
        statics={n:render_static(renderer,cam,s.static) for n,s in models.items()}
        p=statics['J6_parent'];a=p['alpha'].squeeze();D=p['depth'].squeeze()/a.clamp_min(1e-8)
        valid=(a>.5)&torch.isfinite(D)&(D>0);lo,hi=[float(torch.quantile(D[valid],q)) for q in [0.,1.]]
        for number in [2,100,200]:
            t=(number-1)/199;rows=[];values=[]
            for name,scene in models.items():
                static=statics[name];dynamic=renderer(cam,scene.dynamic_at(t),PIPE,scene.centers.new_zeros(3),0.,0.,0.)
                joint=scene.render(renderer,cam,t)
                arrays=[('staticRGB',rgb(static)),('dynamicRGB',rgb(dynamic)),('jointRGB',rgb(joint)),
                    ('staticAlpha',alpha(static)),('dynamicAlpha',alpha(dynamic)),('jointAlpha',alpha(joint)),
                    ('staticD-alpha',depth(static,lo,hi)),('dynamicD-alpha',depth(dynamic,lo,hi)),('jointD-alpha',depth(joint,lo,hi))]
                rows.append((name,arrays));values.append((name,[('jointRGB',rgb(joint))]))
                for label,array in arrays:
                    path=r.OUT/f'c{camid:03d}_t{number:03d}_{name}_{label}_native.png';Image.fromarray(array).save(path)
                    images[path.relative_to(ROOT).as_posix()]=sha(path)
                if number==2:
                    mask=(static['alpha'].squeeze()>.5);expected=static['depth'].squeeze()/static['alpha'].squeeze().clamp_min(1e-8)
                    support.append({'camera':camid,'model':name,'pixels':mask.numel(),'alpha_gt05':int(mask.sum()),
                        'parent_support':int(valid.sum()),'parent_support_now_missing':int((valid&~mask).sum()),
                        'D_alpha_parent_support_MAE_proxy':float((expected[valid]-D[valid]).abs().mean()),
                        'not_independent_geometry_accuracy':True})
            path=r.OUT/f'c{camid:03d}_t{number:03d}_decomposition.png'
            sheet(rows,f'Rubber Duck c{camid} t{number}; fixed view: NO independent GT; black depth = missing; D range {lo:.3f}..{hi:.3f}').save(path)
            images[path.relative_to(ROOT).as_posix()]=sha(path)
            path=r.OUT/f'c{camid:03d}_t{number:03d}_joint_comparison.png'
            sheet([('',[(name,arr[0][1]) for name,arr in values])],f'Rubber Duck camera{camid} time{number}: J6 / direct fusion / old-static control / color-free trained',cell=(640,360)).save(path)
            images[path.relative_to(ROOT).as_posix()]=sha(path)
    for index in [0,24,48,72,96,119]:
        frame=meta['frames'][index];cam=camera(frame,meta,1600);target=r.frame_target(frame)
        observed=(target.permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        rows=[('observed', [('training trajectory RGB',observed)])]
        for name,scene in models.items():rows.append((name,[('joint RGB',rgb(scene.render(renderer,cam,frame['time'])))]))
        path=r.OUT/f'observed_train_index{index:03d}.png';sheet(rows,frame['file_path']+' observed trajectory; no identity/geometry GT',cell=(800,450)).save(path)
        images[path.relative_to(ROOT).as_posix()]=sha(path)
    return {'images':images,'fixed_cameras':camera_receipts,'static_support_proxies':support,
        'missing_black_depth_is_not_zero_error':True}





