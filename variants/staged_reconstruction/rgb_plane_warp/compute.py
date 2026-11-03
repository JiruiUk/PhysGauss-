"""Complete per-view RGB-only inference, including all rejected candidates."""
import numpy as np
import torch
from .core import fit_slopes,propose


def compute_view(depth,images,features,texture,views,k,index,sources,near,far,protocol):
    yy,xx=np.nonzero(depth>0); uv=np.c_[xx,yy]; old=depth[yy,xx]
    slope=fit_slopes(depth,uv,protocol); output=old.copy(); diagnostics={}
    for start in range(0,len(old),64):
        sl=slice(start,start+64); z=torch.tensor(old[sl],device=images.device)
        unchanged,off=propose(None,z)
        assert unchanged is z and not off['accepted'].any()
        out,values=propose(torch.tensor(uv[sl],dtype=torch.float32,device=z.device),z,
            torch.tensor(slope['coefficients'][sl,:2],dtype=torch.float32,device=z.device),
            torch.tensor(slope['slope_admitted'][sl],device=z.device),images,features,texture,views,k,index,sources,near,far,True)
        output[sl]=out.cpu().numpy()
        for name,value in values.items(): diagnostics.setdefault(name,[]).append(value.cpu().numpy())
    saved={name:np.concatenate(values,axis=1 if name.startswith('held_') else 0) for name,values in diagnostics.items()}
    assert np.array_equal(output[~saved['accepted']],old[~saved['accepted']])
    assert np.array_equal(output!=old,saved['accepted'])
    arrays={'uv':uv,'original':old,'updated':output,**slope,**saved}
    counts={'candidates':len(old),'slope_admitted':int(slope['slope_admitted'].sum()),
            'eligible':int(saved['eligible'].sum()),'finite_proposals':int(np.isfinite(saved['proposed']).sum()),
            'accepted':int(saved['accepted'].sum())}
    return arrays,counts
