import numpy as np
from variants.staged_reconstruction.learned_tracks.seeds import proposals


def test_dense_proposals_respect_mask_spacing_and_texture():
    rng=np.random.default_rng(5)
    gray=rng.uniform(0,255,(25,25)).astype(np.float32)
    mask=np.zeros_like(gray); mask[4:21,4:21]=1
    points=proposals(gray,mask)
    assert len(points)>5
    assert all(mask[int(y),int(x)] for x,y in points)
    distances=np.linalg.norm(points[:,None]-points[None],axis=-1)
    distances[np.diag_indices(len(points))]=np.inf
    assert distances.min()>=4
    assert len(proposals(np.zeros_like(gray),mask))==0
