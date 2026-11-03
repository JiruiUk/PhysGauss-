"""CPU ray traced diagnostic, never imported by the RGB reconstruction loader."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ..prepare_data import ROOT, write, sha

DATA = ROOT/'data/derived/clean_static_synthetic/v1'
REPORT = ROOT/'experiments/staged_reconstruction/S1_clean_visibility/synthetic_v1'
OUT = ROOT/'output/staged_reconstruction/S1_clean_visibility/synthetic_v1'


def camera_view(angle, elevation=.32, radius=4.2):
    origin = np.array([radius*np.sin(angle),radius*elevation,radius*np.cos(angle)])
    forward = -origin/np.linalg.norm(origin)
    right = np.cross(forward,[0,1.,0]); right /= np.linalg.norm(right)
    down = np.cross(forward,right)
    rotation = np.stack([right,down,forward])
    view = np.eye(4); view[:3,:3] = rotation; view[:3,3] = -rotation@origin
    return view


def box(origin,rays,lo,hi):
    safe = np.where(np.abs(rays) > 1e-12,rays,1e-12)
    a,b = (np.asarray(lo)-origin)/safe,(np.asarray(hi)-origin)/safe
    near,far = np.minimum(a,b),np.maximum(a,b)
    entry,leave = near.max(-1),far.min(-1)
    hit = (leave >= entry)&(entry > .01)
    axis = near.argmax(-1)
    normal = np.zeros_like(rays)
    normal[np.arange(len(rays)),axis] = -np.sign(rays[np.arange(len(rays)),axis])
    return np.where(hit,entry,np.inf),normal


def sphere(origin,rays,center,radius):
    delta = origin-np.asarray(center); a = (rays*rays).sum(-1); b = rays@delta
    discriminant = b*b-a*(delta@delta-radius*radius)
    depth = (-b-np.sqrt(np.maximum(discriminant,0)))/a
    good = (discriminant >= 0)&(depth > .01)
    normal = (origin+rays*depth[:,None]-center)/radius
    return np.where(good,depth,np.inf),normal


def trace(view,k,width,height,t,glossy=False,static_only=False):
    yy,xx = np.mgrid[:height,:width]
    local = np.c_[xx.ravel(),yy.ravel(),np.ones(width*height)]@np.linalg.inv(k).T
    inverse = np.linalg.inv(view); origin = inverse[:3,3]; rays = local@inverse[:3,:3].T
    # Texture/color is shared across stationary and moving surfaces; no semantic color cue.
    surfaces = [box(origin,rays,[-.8,-.75,-.6],[.8,-.45,.6]),
                box(origin,rays,[-.2,-.45,-.2],[.2,.9,.2]),
                box(origin,rays,[.42,-.45,-.32],[.46,.6,.32])]
    if not static_only:
        surfaces += [sphere(origin,rays,[.8*np.sin(2*np.pi*t),.28,.7*np.cos(2*np.pi*t)],.25),
                     box(origin,rays,[-.45+.6*t,.62,-.7],[-.2+.6*t,.85,-.45])]
    depths = np.stack([s[0] for s in surfaces]); index = depths.argmin(0)
    depth = depths[index,np.arange(len(rays))]; hit = np.isfinite(depth)
    normal = np.stack([s[1] for s in surfaces])[index,np.arange(len(rays))]
    xyz = origin+rays*np.where(hit,depth,0)[:,None]
    texture = .52+.16*np.sin(xyz[:,0]*17+xyz[:,1]*9)+.12*np.sin(xyz[:,2]*23-xyz[:,1]*13)
    color = np.stack([texture*.9,texture*.75+.08,texture*.6+.16],1)
    light = np.array([-.4,.8,.6]); light /= np.linalg.norm(light)
    diffuse = .35+.65*np.maximum(normal@light,0)
    color *= diffuse[:,None]
    if glossy:
        eye = -rays/np.linalg.norm(rays,axis=1,keepdims=True)
        half = eye+light; half /= np.linalg.norm(half,axis=1,keepdims=True)
        color += .6*np.maximum((normal*half).sum(1),0)[:,None]**40
    color[~hit] = 0; normal[~hit] = 0
    return {'rgb':np.clip(color,0,1).reshape(height,width,3),
            'depth':np.where(hit,depth,0).reshape(height,width).astype(np.float32),
            'normal':normal.reshape(height,width,3).astype(np.float32),
            'static_visible':(hit&(index < 3)).reshape(height,width),
            'thin_static_visible':(hit&(index == 2)).reshape(height,width)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--enable-synthetic-audit',action='store_true'); args = parser.parse_args()
    if not args.enable_synthetic_audit: raise SystemExit('Synthetic audit is opt-in')
    assert not DATA.exists(), 'Do not overwrite an existing audit dataset'
    width,height = 256,144; focal = width/(2*np.tan(np.radians(56)/2))
    k = np.array([[focal,0,(width-1)/2],[0,focal,(height-1)/2],[0,0,1.]])
    views = [camera_view(2*np.pi*c/16) for c in range(16)]
    times = np.linspace(0,1,5,endpoint=False)
    # Cartesian camera/time grid prevents the real dataset's orbit/time confound.
    # Each of12 training cameras sees2 times;4 camera poses are held out entirely.
    pairs = [(c,ti) for c in range(16) for ti in range(5)]
    train_pairs = [(c,ti) for c in range(16) if c%4 != 3 for ti in [(2*c)%5,(2*c+2)%5]]
    REPORT.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    panels = []
    for case,glossy in [('diffuse',False),('glossy',True)]:
        directory = DATA/case; train = directory/'train'; private = directory/'evaluation_only'
        (train/'rgb').mkdir(parents=True); (private/'gt').mkdir(parents=True); rows = []
        for c,ti in pairs:
            view,t = views[c],float(times[ti]); observed = trace(view,k,width,height,t,glossy)
            clean = trace(view,k,width,height,t,glossy,static_only=True)
            name = f'c{c:02d}_t{ti}'
            np.savez_compressed(private/'gt'/f'{name}.npz',observed_rgb=observed['rgb'].astype(np.float32),
                static_rgb=clean['rgb'].astype(np.float32),static_depth=clean['depth'],static_normal=clean['normal'],
                observed_static_visible=observed['static_visible'],thin_static_visible=clean['thin_static_visible'])
            if (c,ti) in train_pairs:
                Image.fromarray((observed['rgb']*255).round().astype('uint8')).save(train/'rgb'/f'{name}.png')
                rows.append({'image':f'rgb/{name}.png','world_to_camera':view.tolist(),'time':t})
            if c in [0,4] and ti in [0,2,4]:
                panel = Image.new('RGB',(width*2,height+24),'#161922')
                panel.paste(Image.fromarray((observed['rgb']*255).round().astype('uint8')),(0,24))
                panel.paste(Image.fromarray((clean['rgb']*255).round().astype('uint8')),(width,24))
                ImageDraw.Draw(panel).text((5,5),f'{case} camera{c} t{t:.1f}: RGB | evaluator-only static',fill='white')
                panels.append(np.asarray(panel))
        write(train/'cameras.json',{'width':width,'height':height,'K':k.tolist(),'frames':rows})
        write(private/'cameras.json',{'width':width,'height':height,'K':k.tolist(),
            'frames':[{'case':case,'camera':c,'time_index':ti,'time':float(times[ti]),'world_to_camera':views[c].tolist(),
                       'seen_camera':c%4 != 3,'in_training':(c,ti) in train_pairs} for c,ti in pairs]})
    Image.fromarray(np.concatenate(panels)).save(OUT/'audit_dataset_preview.png')
    write(REPORT/'dataset.json',{'version':1,'cases':['diffuse','glossy'],'train_frames_per_case':len(train_pairs),
        'factorial_camera_time_pairs_per_case':len(pairs),'novel_camera_pairs_per_case':20,
        'train_inputs':'Only opaque RGB, supplied calibrated camera and time. No scene/static labels, depth, normal, alpha or object count.',
        'GT_contract':'All clean static renders/depth/normal/visibility and evaluator camera grid live under evaluation_only. Reconstruction loader never opens these files.',
        'known_limit':'Analytic small controlled RGB diagnostic, no physically simulated deformation/contact; not a PIDG benchmark reproduction.',
        'generator_sha256':sha(Path(__file__)),'input_loader_sha256':sha(Path(__file__).with_name('input.py')),
        'files':{p.relative_to(ROOT).as_posix():sha(p) for p in DATA.rglob('*') if p.is_file()}})
    print('SYNTHETIC_AUDIT_GENERATED',len(train_pairs),'train images/case,80 evaluator camera/time pairs/case',flush=True)


if __name__ == '__main__': main()
