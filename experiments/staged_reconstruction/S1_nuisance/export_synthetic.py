"""Read-only PLY/video export after frozen synthetic evaluation; no training."""
from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw
import torch
from variants.staged_reconstruction.prepare_data import ROOT,read,write,sha
from variants.staged_reconstruction.nuisance_static.prepare import REPORT,RUNS,OUT,verify
from variants.staged_reconstruction.nuisance_static.runtime import calibrated_camera
from variants.staged_reconstruction.nuisance_static.evaluate import panel
from variants.staged_reconstruction.adaptive_static.model import AdaptiveGaussian
from variants.staged_reconstruction.train_static import setup,render_static
from variants.staged_reconstruction.cross_temporal_static.evaluate import camera_hash


def encode(path,frames):
    assert not path.exists(), 'Preserve previous export'
    with path.with_suffix('.encode.log').open('wb') as log:
        process = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-n',
            '-f','rawvideo','-pix_fmt','rgb24','-s','512x336','-r','20','-i','pipe:0','-an',
            '-c:v','libx264','-threads','4','-profile:v','main','-pix_fmt','yuv420p','-crf','18',
            '-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW)
        count = 0
        for frame in frames: process.stdin.write(np.asarray(frame).tobytes()); count += 1
        process.stdin.close(); assert process.wait(timeout=60) == 0
    cap = cv2.VideoCapture(str(path)); decoded,contacts = 0,[]
    while True:
        ok,value = cap.read()
        if not ok: break
        if decoded in [0,4,20,79,120,159]: contacts.append(cv2.cvtColor(value,cv2.COLOR_BGR2RGB))
        decoded += 1
    cap.release(); assert decoded == count == 160
    contact = path.with_suffix('.decoded.png'); Image.fromarray(np.concatenate(contacts)).save(contact)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'decoded_frames':decoded,
        'codec':'H264 Main/yuv420p/faststart','actual_browser_playback_checked':False,
        'content':'Independent synthetic camera/time factor grid; not a real observed camera trajectory or physical motion model.'}


def depth_alpha_panel(packages,gt,label):
    """Evaluation-only visualization; common GT depth range, expectedD/alpha."""
    depth = gt['static_depth']; valid = depth > 0
    lo,hi = float(depth[valid].min()),float(depth[valid].max())
    values = [('GT depth',depth,valid.astype(float))]
    for arm,package in packages:
        alpha = package['alpha'].squeeze().cpu().numpy()
        expected = package['depth'].squeeze().cpu().numpy()/np.maximum(alpha,1e-8)
        assert np.isfinite(expected).all() and np.isfinite(alpha).all()
        values.append((arm,expected,alpha))
    height,width = depth.shape; canvas = Image.new('RGB',(3*width,2*(height+24)),'#161922')
    for col,(name,expected,alpha) in enumerate(values):
        mapped = np.clip((expected-lo)/max(hi-lo,1e-8),0,1)
        color = cv2.cvtColor(cv2.applyColorMap((mapped*255).astype('uint8'),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
        color[alpha <= .5] = 0
        for row,(title,array) in enumerate([(f'{label} {name} D/alpha {lo:.2f}..{hi:.2f}',color),
                (f'{label} {name} alpha 0..1',np.repeat((np.clip(alpha,0,1)*255).astype('uint8')[...,None],3,2))]):
            x,y = col*width,row*(height+24)
            canvas.paste(Image.fromarray(array),(x,y+24)); ImageDraw.Draw(canvas).text((x+3,y+3),title,fill='white')
    return np.asarray(canvas)


@torch.no_grad()
def main():
    verify(); assert (REPORT/'synthetic_evaluation.json').exists()
    assert not (REPORT/'synthetic_exports.json').exists()
    renderer = setup(); torch.set_num_threads(4); cv2.setNumThreads(2)
    protocol = read(REPORT.parent/'protocol.json'); evaluation = read(REPORT/'synthetic_evaluation.json')
    from scene_PIDG.gaussian_model import GaussianModel
    from plyfile import PlyData
    plys,videos,cameras,diagnostics = {},{},{},{}
    for case in protocol['synthetic_cases']:
        models = {}
        for arm in protocol['arms']:
            checkpoint = RUNS/'formal'/case/arm/'checkpoint.pt'
            assert sha(checkpoint) == evaluation['checkpoint_sha256'][checkpoint.relative_to(ROOT).as_posix()]
            state = torch.load(checkpoint,map_location='cpu',weights_only=False)
            model = AdaptiveGaussian(RUNS/case/'initialization.npz'); model.load_payload(state['model']); models[arm] = model
            native = GaussianModel(3); native.active_sh_degree = 3
            for name,p in model.named_parameters(): setattr(native,name,p)
            native.dynamic_mask = torch.zeros(len(model.get_xyz),dtype=torch.bool,device='cuda')
            path = OUT/f'{case}_{arm}_6000.ply'; assert not path.exists(); native.save_ply(str(path))
            vertex = PlyData.read(path)['vertex']; xyz = np.stack([vertex[n] for n in ['x','y','z']],1)
            assert np.array_equal(xyz,model.get_xyz.cpu().numpy())
            assert all(np.isfinite(vertex[n]).all() for n in vertex.data.dtype.names)
            plys[case+'/'+arm] = {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
                'vertices':len(xyz),'checkpoint_positions_exact':True,'all_fields_finite':True}
        private = ROOT/protocol['synthetic_data']/case/'evaluation_only'; meta = read(private/'cameras.json')
        cache,depth_panels = {},[]
        for row in meta['frames']:
            if row['time_index'] != 0: continue
            cam = calibrated_camera(np.asarray(row['world_to_camera']),np.asarray(meta['K']),meta['width'],meta['height'])
            fingerprint = camera_hash(cam); values,packages = [],[]
            for arm,model in models.items():
                p = render_static(renderer,cam,model)
                values.append((arm.replace('full_RGB_','')+' static',p['render'].clamp(0,1).permute(1,2,0).cpu().numpy()))
                packages.append((arm.replace('full_RGB_',''),p))
                assert camera_hash(cam) == fingerprint
            cache[row['camera']] = values; cameras[case+'/'+str(row['camera'])] = fingerprint
            if row['camera'] in [3,24,31]:
                with np.load(private/'gt'/f"c{row['camera']:02d}_t0.npz") as gt:
                    depth_panels.append(depth_alpha_panel(packages,gt,f'{case} c{row["camera"]}'))
        path = OUT/f'{case}_depth_alpha.png'; assert not path.exists()
        Image.fromarray(np.concatenate(depth_panels)).save(path)
        diagnostics[case] = {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
            'contract':'ExpectedD/alpha, black if alpha<=.5; shared per-camera GT depth color range; not all invisible surfaces.'}
        def frames():
            for row in meta['frames']:
                with np.load(private/'gt'/f"c{row['camera']:02d}_t{row['time_index']}.npz") as data:
                    values = [('Observed synthetic RGB',data['observed_rgb']),('Static GT',data['static_rgb'])]+cache[row['camera']]
                    yield panel(values,f"{case} c{row['camera']} t{row['time']:.1f}")
        videos[case] = encode(OUT/f'{case}_camera_time_comparison.mp4',frames())
    for p,h in evaluation['checkpoint_sha256'].items(): assert sha(ROOT/p) == h
    verify(); write(REPORT/'synthetic_exports.json',{'passed':True,'PLY':plys,'videos':videos,
        'camera_fingerprints':cameras,'depth_alpha':diagnostics,'export_source_sha256':sha(Path(__file__)),'training_or_checkpoints_modified':False})
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>S1-N synthetic gate</title>'
        '<style>body{background:#171923;color:white;font:17px system-ui;max-width:1100px;margin:auto}img,video{max-width:100%}</style>'
        '<h1>S1-N independent synthetic evaluation</h1><p>GT is evaluator-only. Nuisance planes are per-image appearance explanations, not identified physical objects.</p>'
        '<h2>Diffuse: static GT / observed / control / nuisance-static</h2><img src="diffuse_static_comparison.png">'
        '<h2>Glossy</h2><img src="glossy_static_comparison.png">'
        '<h2>Expected depthD/alpha and alpha: c3/c24/c31</h2><img src="diffuse_depth_alpha.png"><img src="glossy_depth_alpha.png">'
        '<h2>Train-view decomposition</h2><img src="diffuse_train_decomposition.png"><img src="glossy_train_decomposition.png">'
        '<h2>Camera/time grid, not physical motion prediction</h2><video controls src="diffuse_camera_time_comparison.mp4"></video>'
        '<video controls src="glossy_camera_time_comparison.mp4"></video>',encoding='utf-8')
    print('NUISANCE_SYNTHETIC_EXPORTED',videos,flush=True)


if __name__ == '__main__': main()
