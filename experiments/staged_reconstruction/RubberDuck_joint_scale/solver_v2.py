"""Numerical repair only: same convex Huber/L2 objective, exact dense TR solve."""
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from variants.staged_reconstruction.i10_tracking.core import sample


def fit(bank,norms,raw,p,half=None,enabled=False,original=None):
    if not enabled:return original
    n=len(bank);frames=[];values=[];inverse=[]
    for i,rows in enumerate(bank):
        chosen=[q for q in rows if q['split']!='check' and (half is None or q['split']==f'fit{half}')]
        med,spread=norms[i]
        if not chosen or spread<=1e-12:continue
        frames.extend([i]*len(chosen))
        values.extend((sample(raw[i],np.array([q['uv'] for q in chosen]))-med)/spread)
        inverse.extend(1/np.array([q['depth'] for q in chosen]))
    if not inverse:return dict(success=False,reason='no_fit_anchors',coefficients=None)
    frames=np.array(frames);x=np.array(values,float);y=np.array(inverse);scale=float(np.median(y))
    count=np.bincount(frames,minlength=n);weight=1/count[frames];nd=len(y)
    matrix=np.zeros((nd+2*(n-2)+2*n,2*n))
    matrix[np.arange(nd),frames*2]=x/y;matrix[np.arange(nd),frames*2+1]=1/y
    for i in range(n-2):
        for c in [0,1]:matrix[nd+i*2+c,[i*2+c,(i+1)*2+c,(i+2)*2+c]]=np.array([1.,-2.,1.])*np.sqrt(p['smoothness'])/scale
    matrix[nd+2*(n-2):]=np.eye(2*n)*np.sqrt(p['ridge'])/scale
    target=np.r_[np.ones(nd),np.zeros(matrix.shape[0]-nd)]
    def loss(z):
        rho=np.vstack([z,np.ones_like(z),np.zeros_like(z)])
        indices=np.flatnonzero(z[:nd]>1)
        rho[0,indices]=2*np.sqrt(z[indices])-1
        rho[1,indices]=1/np.sqrt(z[indices]);rho[2,indices]=-.5/z[indices]**1.5
        rho[:,:nd]*=weight
        return rho
    with threadpool_limits(limits=4):
        result=least_squares(lambda v:matrix@v-target,np.tile([scale*.1,scale],n),
            jac=lambda v:matrix.copy(),bounds=(np.tile([p['minimum_slope'],-np.inf],n),np.full(2*n,np.inf)),
            loss=loss,f_scale=p['huber_scale'],x_scale='jac',tr_solver='exact',
            max_nfev=p['maximum_iterations'],ftol=p['solver_tolerance'],
            xtol=p['solver_tolerance'],gtol=p['solver_tolerance'])
    normalized=result.x.reshape(n,2);coefficients=[]
    for (a,b),(med,spread) in zip(normalized,norms):
        coefficients.append([a/spread,b-a*med/spread] if spread>1e-12 else [np.nan,np.nan])
    return dict(success=bool(result.success),reason=str(result.message),coefficients=coefficients,
        normalized_coefficients=normalized.tolist(),scale=scale,fit_observations=nd,
        iterations=int(result.nfev),cost=float(result.cost),optimality=float(result.optimality),
        coefficients_model='positive normalized affine with fixed temporal second-difference prior',
        numerical_solver='dense exact trust region; Jacobian column scaling; same objective/bounds/start/tolerances/budget')
