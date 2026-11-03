"""All parent candidates saved, including absent/invalid source hypotheses."""
import numpy as np
import torch
from .core import propose

SOURCE_FIELDS={'source_uv','source_query_valid','source_original_depth','source_slopes','raw_candidate_depth',
    'raw_candidate_slopes','candidate_valid','source_projected_hit','candidate_fit','held_old','held_new'}


def compute_view(parent,anchor_maps,images,features,texture,views,k,index,sources,near,far):
    uv,old=parent['uv'],parent['updated']; output=old.copy(); diagnostics={}
    for start in range(0,len(old),256):
        sl=slice(start,start+256); z=torch.tensor(old[sl],device=images.device)
        off,mask=propose(None,z); assert off is z and not mask['accepted'].any()
        out,values=propose(torch.tensor(uv[sl],dtype=torch.float32,device=z.device),z,
            torch.tensor(parent['coefficients'][sl,:2],dtype=torch.float32,device=z.device),anchor_maps,
            images,features,texture,views,k,index,sources,near,far,True)
        output[sl]=out.cpu().numpy()
        for name,value in values.items(): diagnostics.setdefault(name,[]).append(value.cpu().numpy())
    saved={name:np.concatenate(values,axis=1 if name in SOURCE_FIELDS else 0) for name,values in diagnostics.items()}
    assert np.array_equal(output[~saved['accepted']],old[~saved['accepted']]) and np.array_equal(output!=old,saved['accepted'])
    arrays={'uv':uv,'original':old,'updated':output,'parent_slope_admitted':parent['slope_admitted'],**saved}
    counts={'candidates':len(old),'source_queries_admitted':int(saved['source_query_valid'].sum()),
        'valid_source_hypotheses':int(saved['candidate_valid'].sum()),'eligible':int(saved['eligible'].sum()),
        'eligible_without_target_slope':int((saved['eligible']&~parent['slope_admitted']).sum()),
        'accepted':int(saved['accepted'].sum()),'accepted_without_target_slope':int((saved['accepted']&~parent['slope_admitted']).sum())}
    return arrays,counts
