"""CPU camera hypotheses and depth-free checks; no fitting or model imports."""
import numpy as np


def w2c(c2w,mode='legacy'):
    matrix=np.linalg.inv(np.asarray(c2w,dtype=float))
    if mode=='metadata_cv':return np.diag([1.,-1.,-1.,1.])@matrix
    if mode!='legacy':raise ValueError(mode)
    r=-matrix[:3,:3].T;r[:,0]=-r[:,0]
    out=np.eye(4);out[:3,:3]=r.T;out[:3,3]=-matrix[:3,3]
    return out


def intrinsic(angle,mode='legacy',width=400,height=225):
    f=width/(2*np.tan(angle/2))
    if mode=='metadata':fx=fy=f
    elif mode=='legacy':fx=f*width/height;fy=f*height/width
    else:raise ValueError(mode)
    return np.array([[fx,0,(width-1)/2],[0,fy,(height-1)/2],[0,0,1.]])


def project(points,view,k):
    p=np.asarray(points)@view[:3,:3].T+view[:3,3]
    return (p@k.T)[:,:2]/p[:,2:3],p[:,2]


def fundamental(first,second,k1,k2):
    relative=second@np.linalg.inv(first);r=relative[:3,:3];t=relative[:3,3]
    if np.linalg.norm(t)<1e-8:return None
    x,y,z=t;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    f=np.linalg.inv(k2).T@cross@r@np.linalg.inv(k1)
    return f/np.linalg.norm(f)


def sampson(first,second,f):
    if f is None:return np.full(len(first),np.nan)
    f=f/np.linalg.norm(f);x=np.c_[first,np.ones(len(first))];y=np.c_[second,np.ones(len(second))]
    fx=x@f.T;fy=y@f;residual=np.sum(y*fx,axis=1)
    denominator=np.sum(fx[:,:2]**2,axis=1)+np.sum(fy[:,:2]**2,axis=1)
    return np.abs(residual)/np.sqrt(np.maximum(denominator,1e-30))


def motion_split(source_world,target_world,source_view,target_view,k):
    a,_=project(source_world,source_view,k);b,_=project(source_world,target_view,k);c,_=project(target_world,target_view,k)
    return c-a,b-a,c-b
