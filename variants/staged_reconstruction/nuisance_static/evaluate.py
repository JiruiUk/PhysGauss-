"""Independent static-GT synthetic evaluation; never used by the trainer."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
import torch
from ..prepare_data import ROOT,read,write,sha
from ..adaptive_static.model import AdaptiveGaussian
from ..adaptive_static.prepare import world_points
from ..synthetic_audit.evaluate import score
from ..synthetic_audit.input import load
from ..train_static import setup,render_static
from .runtime import calibrated_camera,layers
from .core import NuisancePlane
from .prepare import REPORT,RUNS,OUT,verify


def normal_error(expected,alpha,gt_depth,gt_normal,view,k):
    height,width = expected.shape; yy,xx = np.mgrid[:height,:width]
    xyz = world_points(np.c_[xx.ravel(),yy.ravel()],expected.ravel(),view,k).reshape(height,width,3)
    normal = np.cross(xyz[1:-1,2:]-xyz[1:-1,:-2],xyz[2:,1:-1]-xyz[:-2,1:-1])
    length = np.linalg.norm(normal,axis=2); normal /= np.maximum(length[...,None],1e-12)
    valid = (gt_depth[1:-1,1:-1] > 0)&(length > 1e-8)
    for ay,ax in [(slice(1,-1),slice(1,-1)),(slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),(slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))]:
        valid &= (alpha[ay,ax] > .95)&(gt_depth[ay,ax] > 0)
        valid &= np.abs((gt_normal[1:-1,1:-1]*gt_normal[ay,ax]).sum(2)) > .99
    angle = np.degrees(np.arccos(np.clip(np.abs((normal*gt_normal[1:-1,1:-1]).sum(2)),0,1)))
    return float(angle[valid].mean()) if valid.any() else None,int(valid.sum())


def image_metrics(package,gt,view,k):
    predicted = package['render'].clamp(0,1).permute(1,2,0).cpu().numpy()
    alpha = package['alpha'].squeeze().cpu().numpy()
    accumulated = package['depth'].squeeze().cpu().numpy()
    expected = accumulated/np.maximum(alpha,1e-8)
    surface = gt['static_depth'] > 0; thin = gt['thin_static_visible']; background = ~surface
    error = (predicted-gt['static_rgb'])**2
    valid_depth = surface&(alpha > .5)&np.isfinite(expected)
    angle,pixels = normal_error(expected,alpha,gt['static_depth'],gt['static_normal'],view,k)
    return {'clean_GT_PSNR':float(-10*np.log10(max(float(error.mean()),1e-12))),
        'static_region_PSNR':float(-10*np.log10(max(float(error[surface].mean()),1e-12))),
        'static_RGB_spill':float(np.abs(predicted[background]).mean()),
        'static_alpha_recall':float((alpha[surface] > .5).mean()),
        'thin_alpha_recall':float((alpha[thin] > .5).mean()) if thin.any() else None,
        'background_alpha':float(alpha[background].mean()),
        'expected_depth_MAE':float(np.abs(expected[valid_depth]-gt['static_depth'][valid_depth]).mean()) if valid_depth.any() else None,
        'normal_unsigned_error_deg':angle,'normal_evaluated_pixels':pixels},predicted


def panel(images,label):
    height,width = images[0][1].shape[:2]; canvas = Image.new('RGB',(width*2,(height+24)*2),'#161922')
    for i,(name,value) in enumerate(images):
        x,y = (i%2)*width,(i//2)*(height+24)
        canvas.paste(Image.fromarray((np.clip(value,0,1)*255).round().astype('uint8')),(x,y+24))
        ImageDraw.Draw(canvas).text((x+4,y+4),label+' '+name,fill='white')
    return np.asarray(canvas)


@torch.no_grad()
def main():
    verify(); assert not (REPORT/'synthetic_evaluation.json').exists(), 'Preserve completed evaluation'
    torch.set_num_threads(4); cv2.setNumThreads(2); renderer = setup(); protocol = read(REPORT.parent/'protocol.json')
    dataset_receipt = read(ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v2/dataset.json')
    for p,h in dataset_receipt['files'].items(): assert sha(ROOT/p) == h,p
    results,gate,images,hashes = {},{}, {},{}
    for case in protocol['synthetic_cases']:
        models,states = {},{}
        for arm in protocol['arms']:
            directory = RUNS/'formal'/case/arm; path = directory/'checkpoint.pt'
            audit = read(REPORT/f'formal_{case}_{arm}_audit.json')
            assert audit['passed'] and audit['iteration'] == 6000 and audit['checkpoint_sha256'] == sha(path)
            state = torch.load(path,map_location='cpu',weights_only=False)
            model = AdaptiveGaussian(RUNS/case/'initialization.npz'); model.load_payload(state['model'])
            models[arm],states[arm] = model,state; hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        sequences = [[json.loads(line)['view'] for line in (RUNS/'formal'/case/arm/'sampling.jsonl').read_text().splitlines()] for arm in protocol['arms']]
        assert sequences[0] == sequences[1] and len(sequences[0]) == 6000
        directory = ROOT/protocol['synthetic_data']/case/'evaluation_only'; meta = read(directory/'cameras.json')
        k = np.asarray(meta['K']); reference,thin_reference,rows,panels = [],[],[],[]
        for row in meta['frames']:
            if row['time_index'] != 0: continue
            view = np.asarray(row['world_to_camera']); cam = calibrated_camera(view,k,meta['width'],meta['height'])
            with np.load(directory/'gt'/f"c{row['camera']:02d}_t0.npz") as data: gt = {key:data[key].copy() for key in data.files}
            yy,xx = np.nonzero(gt['static_depth'] > 0)
            xyz = world_points(np.c_[xx,yy],gt['static_depth'][yy,xx],view,k)
            reference.append(xyz); thin_reference.append(xyz[gt['thin_static_visible'][yy,xx]])
            record = {'camera':row['camera'],'novel_camera':not row['seen_camera'],'models':{}}
            values = [('Static GT',gt['static_rgb']),('Observed RGB has motion',gt['observed_rgb'])]
            for arm,model in models.items():
                package = render_static(renderer,cam,model); metrics,predicted = image_metrics(package,gt,view,k)
                record['models'][arm] = metrics; values.append((arm.replace('full_RGB_',''),predicted))
                if row['camera'] in [0,3,8,15,24,31]:
                    path = OUT/f'{case}_{arm}_camera{row["camera"]:02d}.png'
                    Image.fromarray((predicted*255).round().astype('uint8')).save(path); images[path.relative_to(ROOT).as_posix()] = sha(path)
            if row['camera'] in [0,3,8,15,24,31]: panels.append(panel(values,f"{case} c{row['camera']}"))
            rows.append(record)
        path = OUT/f'{case}_static_comparison.png'; Image.fromarray(np.concatenate(panels)).save(path)
        images[path.relative_to(ROOT).as_posix()] = sha(path)
        def unique(x):
            _,indices = np.unique(np.floor(x/.015).astype(int),axis=0,return_index=True); return x[indices]
        reference,thin_reference = unique(np.concatenate(reference)),unique(np.concatenate(thin_reference))
        aggregate = {}
        for arm,model in models.items():
            active = (model.get_opacity[:,0] >= .1).cpu().numpy()
            geometry = score(model.get_xyz.cpu().numpy()[active],reference,thin_reference,.05)
            split = {}
            for label,subset in [('all',rows),('novel_camera',[r for r in rows if r['novel_camera']])]:
                keys = subset[0]['models'][arm].keys(); split[label] = {}
                for key in keys:
                    v = [r['models'][arm][key] for r in subset if r['models'][arm][key] is not None]
                    split[label][key] = float(np.mean(v)) if v else None
            aggregate[arm] = {'geometry_active_opacity_ge_01':geometry,'images':split}
        # Render actual train-view decomposition, separately from independent static GT.
        training = load(ROOT/protocol['synthetic_data']/case/'train'); decomposition = []
        state = states['full_RGB_nuisance']; static = models['full_RGB_nuisance']
        for index in [0,24,48,72]:
            im,view = training['rgb'][index],training['views'][index]
            plane = NuisancePlane(view,training['K'],im,count=1000,seed=20261103+index)
            payload = {n.split('.',1)[1]:v for n,v in state['nuisance'].items() if n.startswith(str(index)+'.')}
            plane.load_state_dict(payload); cam = calibrated_camera(view,training['K'],im.shape[1],im.shape[0])
            package = layers(renderer,cam,static,plane,True)
            cpu = lambda t:t.clamp(0,1).permute(1,2,0).cpu().numpy()
            decomposition.append(panel([('Observed train RGB',im),('Shared static',cpu(package['static']['render'])),
                ('Nuisance premultiplied',cpu(package['nuisance']['render'])),('Composite',cpu(package['composite']))],f'{case} train{index}'))
        Image.fromarray(np.concatenate(decomposition)).save(OUT/f'{case}_train_decomposition.png')
        control,nuisance = [aggregate[a] for a in protocol['arms']]; c,n = control['images']['novel_camera'],nuisance['images']['novel_camera']
        cg,ng = control['geometry_active_opacity_ge_01'],nuisance['geometry_active_opacity_ge_01']
        limit = protocol['synthetic_promotion_gate']
        details = {'static_clean_GT_PSNR_gain_dB':n['clean_GT_PSNR']-c['clean_GT_PSNR'],
            'visible_surface_coverage_loss_percentage_points':100*(cg['surface_coverage']-ng['surface_coverage']),
            'thin_surface_coverage_loss_percentage_points':100*(cg['thin_surface_coverage']-ng['thin_surface_coverage']),
            'static_RGB_spill_reduction_fraction':1-n['static_RGB_spill']/max(c['static_RGB_spill'],1e-12)}
        passed = (details['static_clean_GT_PSNR_gain_dB'] >= limit['static_clean_GT_PSNR_gain_min_dB'] and
                  details['visible_surface_coverage_loss_percentage_points'] <= limit['visible_surface_coverage_loss_max_percentage_points'] and
                  details['thin_surface_coverage_loss_percentage_points'] <= limit['thin_surface_coverage_loss_max_percentage_points'] and
                  details['static_RGB_spill_reduction_fraction'] >= limit['static_RGB_spill_reduction_min_fraction'])
        gate[case] = {'numerical_passed':bool(passed),**details}; results[case] = {'aggregate':aggregate,'frames':rows,'matched_6000_camera_sequence':True}
    assert all(sha(ROOT/p) == h for p,h in hashes.items()); verify()
    write(REPORT/'synthetic_evaluation.json',{'results':results,'checkpoint_sha256':hashes,'images':images,
        'independent_static_GT_evaluation':True,'training_GT_read':False,'expected_depth_contract':'native accumulatedD divided by alpha; evaluate only alpha>.5',
        'normal_contract':'Unsigned normals from expected-depth finite differences, alpha>.95 and constant GT face neighborhood',
        'geometry_contract':'Active opacity>=.1 model centers to camera-visible GT,0.05 tolerance; thin thickness0.04 is below tolerance, not thickness proof',
        'primary_image_gate':'8 genuinely novel cameras; static-only render, no nuisance fitted to held-out images',
        'dataset_file_hashes_verified':True,'evaluation_source_sha256':sha(Path(__file__))})
    write(REPORT/'synthetic_gate.json',{'numerical_passed':all(v['numerical_passed'] for v in gate.values()),
        'cases':gate,'manual_review_required':True,'passed':False,'real_training_authorized_by_gate':False})
    print('NUISANCE_SYNTHETIC_EVALUATED',gate,flush=True)


if __name__ == '__main__': main()
