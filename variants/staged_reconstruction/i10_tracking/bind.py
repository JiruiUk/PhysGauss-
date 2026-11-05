"""Immutable I10-descendant Gaussian trajectory evidence; no optimization."""
import argparse
from collections import Counter
from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw
import torch
from .io import ROOT,BASE,REPORT,RUNS,OUT,DATA,read,write,sha,lock,verify,stopped
from .core import project_world,stationary_test,bind_track
from output.short_clip_diagnostic.camera_math import w2c,intrinsic


@torch.no_grad()
def parent_trajectories():
    from ..i10_joint.runtime import load
    from ..rubber_joint.runtime import assert_equivalent,bindings
    from ..joint_rgb.core import restore_rng,rng_state
    from ..state_recovery.runtime import restore_graphs,graph_state
    from ..train_static import camera
    p=read(BASE/'protocol.json');assert sha(ROOT/p['parent'])==p['parent_sha256']
    renderer,scene,opts,teacher,parent=load('joint_depth_trust')
    state=torch.load(ROOT/p['parent'],map_location='cpu',weights_only=False)
    scene.load_state_dict(state['model'],strict=True)
    for opt,value in zip(opts,state['optimizers']):opt.load_state_dict(value)
    assert bindings(scene,opts)==state['bindings']
    restore_graphs(scene,state['physics_graphs']);restore_rng(state)
    before_rng=deepcopy(rng_state())
    meta=read(DATA/'transforms_train.json');frames=meta['frames']
    k=intrinsic(meta['camera_angle_x'],'metadata',400,225)
    views=np.array([w2c(f['transform_matrix'],'metadata_cv') for f in frames])
    xyz=[];maximum=0
    for index,frame in enumerate(frames):
        assert not stopped()
        points=scene.dynamic_at(frame['time']).get_xyz
        xyz.append(points.cpu().numpy())
        cam=camera(frame,meta,400)
        h=torch.cat([points,points.new_ones((len(points),1))],1)@cam.full_proj_transform
        native=((h[:,:2]/h[:,3:]+1)*points.new_tensor([400,225])-1)/2
        cpu,z=project_world(xyz[-1][None],views[index:index+1],k)
        valid=(z[0]>.01)&(np.abs(cpu[0]).max(-1)<10000)
        maximum=max(maximum,float(np.abs(native.cpu().numpy()[valid]-cpu[0,valid]).max()))
    assert maximum<=p['camera_projection_parity_max_px'],maximum
    assert_equivalent(scene.state_dict(),state['model'],0,0)
    for opt,value in zip(opts,state['optimizers']):assert_equivalent(opt.state_dict(),value,0,0)
    assert_equivalent(graph_state(scene),state['physics_graphs'],0,0)
    assert_equivalent(rng_state(),before_rng,0,0)
    return np.stack(xyz),views,k,dict(maximum_projection_error_px=maximum,
        model_two_Adam_binding_physics_cache_and_RNG_unchanged=True,parent_sha256=p['parent_sha256'])


def resolve_conflicts(rows,config):
    conflicts=set()
    accepted=[r for r in rows if r.get('binding',{}).get('accepted')]
    for i,a in enumerate(accepted):
        for b in accepted[i+1:]:
            if a['binding']['row']!=b['binding']['row']:continue
            low=max(a['start'],b['start']);high=min(a['start']+12,b['start']+12)
            if high<=low:continue
            x=np.asarray(a['uv'])[low-a['start']:high-a['start']]
            y=np.asarray(b['uv'])[low-b['start']:high-b['start']]
            if np.linalg.norm(x-y,axis=-1).max()>1.5:conflicts.update([a['id'],b['id']])
    for row in rows:
        if row['id'] in conflicts:row['binding'].update(accepted=False,conflict=True)
    return len(conflicts)


def calculate(tracks,xyz,views,k,config):
    predicted,depth=project_world(xyz,views,k);rows=[]
    for track in tracks:
        start=track['start'];sl=slice(start,start+12)
        stationary=stationary_test(track['uv'],views[sl],k,config)
        row=dict(track,stationarity=stationary)
        if stationary['label']=='motion_candidate':
            row['binding']=bind_track(track['uv'],xyz[sl],predicted[sl],depth[sl],config)
        rows.append(row)
    conflicts=resolve_conflicts(rows,config)
    accepted=[r for r in rows if r.get('binding',{}).get('accepted')]
    counts=Counter(r['band'] for r in accepted)
    unique=len({r['binding']['row'] for r in accepted})
    gate=len(accepted)>=config['minimum_bindings'] and min(counts.get(i,0) for i in range(3))>=config['minimum_bindings_per_third'] and unique>=config['minimum_distinct_Gaussian_rows']
    summary=dict(consistent_tracks=len(tracks),stationarity=dict(Counter(r['stationarity']['label'] for r in rows)),
        bound_tracks=len(accepted),bound_by_band={str(i):counts.get(i,0) for i in range(3)},
        distinct_Gaussian_rows=unique,conflicts=conflicts,evidence_gate_passed=bool(gate),optimizer_updates=0)
    return dict(rows=rows,summary=summary)


def review_images(result,xyz,views,k,config):
    predicted,_=project_world(xyz,views,k);meta=read(DATA/'transforms_train.json')
    for start in config['fixed_review_starts']:
        rows=[r for r in result['rows'] if r['start']==start]
        canvas=Image.new('RGB',(1600,741),'#151515')
        for t in range(12):
            frame=meta['frames'][start+t]
            rgba=np.asarray(Image.open(DATA/(frame['file_path']+'.png')).convert('RGBA').resize((400,225),Image.Resampling.BILINEAR))
            tile=Image.fromarray(np.uint8(rgba[...,:3].astype(float)*rgba[...,3:4]/255));draw=ImageDraw.Draw(tile)
            for row in rows:
                x,y=row['uv'][t];label=row['stationarity']['label']
                color={'motion_candidate':'#ffb020','stationary_compatible':'#20baff','unknown':'#aaaaaa'}[label]
                draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
                binding=row.get('binding')
                if binding:
                    bx,by=predicted[start+t,binding['row']]
                    if np.isfinite([bx,by]).all():
                        draw.line([(x,y),(float(bx),float(by))],fill='#30ff50' if binding['accepted'] else '#ff4040',width=1)
            px,py=(t%4)*400,(t//4)*247;canvas.paste(tile,(px,py+22))
            ImageDraw.Draw(canvas).text((px+3,py+4),f'train {start+t} cyan=static-compatible orange=motion',fill='white')
        canvas.save(OUT/f'binding_{start:03d}.png')


def run(replay=False):
    with lock():
        verify();torch.set_num_threads(4);p=read(BASE/'protocol.json')
        if not replay:assert not (REPORT/'bindings.json').exists(),'Preserve completed binding'
        xyz,views,k,audit=parent_trajectories()
        if replay:
            saved=np.load(RUNS/'parent_trajectories.npz')
            for name,value in dict(xyz=xyz,views=views,k=k).items():assert np.array_equal(saved[name],value),name
        else:np.savez_compressed(RUNS/'parent_trajectories.npz',xyz=xyz,views=views,k=k)
        result=calculate(read(REPORT/'tracks.json')['tracks'],xyz,views,k,p)
        if replay:
            assert result==read(REPORT/'bindings.json')
            write(REPORT/'independent_process_replay.json',dict(passed=True,all_parent_trajectories_and_decisions_exact=True,**audit))
        else:
            write(REPORT/'bindings.json',result);write(REPORT/'parent_audit.json',audit)
            review_images(result,xyz,views,k,p)
            write(RUNS/'state.json',dict(status='evidence_complete_pending_review',pending=True,**result['summary']))
        verify();print('PERSISTENT_3D_BINDING',result['summary'],'replay',replay,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-3d-binding',action='store_true');parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    if not args.enable_3d_binding:raise SystemExit('3D binding is opt-in')
    run(args.replay)

if __name__=='__main__':main()
