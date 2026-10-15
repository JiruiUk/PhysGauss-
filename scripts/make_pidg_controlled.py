"""Manufactured adjacent elastic regions with known labels, F, stress and body force.

This is a synthetic engineering fixture, never an official PIDG benchmark result.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

PROJECT=Path(__file__).resolve().parents[1]


def manufacture(destination):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    grid=np.linspace(-.3,.3,4)
    X=np.array([(x,y,z) for x in np.r_[np.linspace(-.45,-.05,4),np.linspace(.05,.45,4)]
                for y in grid for z in grid],dtype=np.float64)
    labels=(X[:,0]>0).astype(np.int64);mu=np.array([1.,2.]);kappa=np.array([10.,20.])
    colors=np.array([[210,65,45],[45,110,210]],dtype=np.uint8)[labels]
    def configuration(t):
        pressure=.6*np.sin(2*np.pi*t)
        lo=np.full(2,.8);hi=np.full(2,1.3)
        for _ in range(64):
            stretch=(lo+hi)/2
            normal=(2*mu/3)*stretch**(-5/3)*(stretch**2-1)+kappa*np.log(stretch)/stretch
            lo=np.where(normal<pressure,stretch,lo);hi=np.where(normal>=pressure,stretch,hi)
        stretch=(lo+hi)/2
        F=np.repeat(np.eye(3)[None],len(X),axis=0);F[:,0,0]=stretch[labels]
        x=np.einsum('nij,nj->ni',F,X);x[:,0]+=.08*np.sin(2*np.pi*t)
        return x,F
    reference=json.loads((PROJECT/'experiments/controlled_camera.json').read_text())
    camera=np.array(reference['transform_matrix']);fov=reference['camera_angle_x']
    matrix=np.linalg.inv(camera);R=-matrix[:3,:3].T;R[:,0]*=-1;T=-matrix[:3,3]
    size=128;focal=size/(2*np.tan(fov/2))
    times=np.r_[np.linspace(0,.7,12),np.linspace(.8,1.,4)]
    all_x=[];all_F=[];all_stress=[];all_force=[]
    (destination/'mask').mkdir()
    for split,indices in [('train',range(12)),('test',range(12,16))]:
        (destination/split).mkdir();frames=[]
        for index in indices:
            t=times[index];x,F=configuration(t);view=x@R+T
            assert (view[:,2]>0).all()
            xy=view[:,:2]/view[:,2,None]*focal+size/2
            image=Image.new('RGBA',(size,size),(0,0,0,0));draw=ImageDraw.Draw(image)
            for point in np.argsort(view[:,2])[::-1]:
                u,v=xy[point];draw.ellipse((u-2,v-2,u+2,v+2),fill=tuple(colors[point])+(255,))
            stem=f'{index:04d}';image.save(destination/split/f'{stem}.png')
            if split=='train':image.getchannel('A').save(destination/'mask'/f'{stem}.png')
            frames.append({'file_path':f'{split}/{stem}','time':float(t),'transform_matrix':camera.tolist()})
            J=np.linalg.det(F);B=F@F.transpose(0,2,1)
            dev=B-np.trace(B,axis1=1,axis2=2)[:,None,None]/3*np.eye(3)
            stress=(mu[labels]*J**(-5/3))[:,None,None]*dev+(kappa[labels]*np.log(J)/J)[:,None,None]*np.eye(3)
            h=1e-3;acceleration=(configuration(t+h)[0]-2*x+configuration(t-h)[0])/h**2
            all_x.append(x);all_F.append(F);all_stress.append(stress);all_force.append(acceleration)
        (destination/f'transforms_{split}.json').write_text(json.dumps({'camera_angle_x':fov,'frames':frames}),encoding='utf-8')
    with (destination/'points3d.ply').open('w',encoding='ascii') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(X)}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for point,color in zip(X,colors):stream.write(' '.join(map(str,[*point,0.,0.,0.,*map(int,color)]))+'\n')
    np.save(destination/'material_ids.npy',labels)
    np.savez(destination/'ground_truth.npz',X=X,times=times,x=np.array(all_x),F=np.array(all_F),stress=np.array(all_stress),
             force=np.array(all_force),material_ids=labels,mu=mu,kappa=kappa,density=np.array(1.))
    audit={'scene':'controlled_two_material','train_count':12,'test_count':4,'missing':[],'shape_mismatches':[],
           'controlled_synthetic':True,'GT':'Manufactured two adjacent elastic regions; labels and physical parameters are prescribed, not inferred.',
           'future_split':{'last_training_time':.7,'first_test_time':.8},
           'force_definition':'rho*a, rho=1; interior stress divergence zero; equal x traction at the planar interface; acceleration uses central differences h=0.001.',
           'images':'Simple projected colored point discs, not photorealistic simulation or official PIDG images.'}
    (destination/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    files={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()}
    (destination/'download_manifest.json').write_text(json.dumps({'origin':'local manufactured engineering fixture','files':files},indent=2),encoding='utf-8')
    return destination


if __name__=='__main__':print(manufacture(PROJECT/'data/PIDG_controlled/two_material'))
