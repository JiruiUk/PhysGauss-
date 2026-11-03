"""Recompute calibration comparison on shared train-only static track candidates."""
import numpy as np
from .prepare_data import ROOT, REPORT, DATA, read, write
from output.short_clip_diagnostic.camera_math import w2c, intrinsic, project
from variants.transfer_integrity.static_audit import triangulate
from pathlib import Path


def main():
    meta=read(DATA/'transforms_train.json')
    frames={int(Path(f['file_path']).stem):f for f in meta['frames']}
    tracks=read(ROOT/'experiments/transfer_integrity/static_image_audit.json')['tracks'];out={}
    for name,mode,kmode in [('legacy','legacy','legacy'),('metadata','metadata_cv','metadata')]:
        k=intrinsic(meta['camera_angle_x'],kmode)
        views={n:w2c(f['transform_matrix'],mode) for n,f in frames.items()}
        matrices={n:k@v[:3] for n,v in views.items()};xyz=[];errors=[]
        for track in tracks:
            observations=list(zip(track['frames'],track['uv']))
            x=triangulate(observations[::2],matrices);xyz.append(x)
            for n,uv in observations[1::2]:
                pred,depth=project(x[None],views[n],k)
                errors.append(float(np.linalg.norm(pred[0]-uv)) if depth[0]>.2 else 400.)
        xyz=np.asarray(xyz)
        out[name]={'held_mean':float(np.mean(errors)),'held_median':float(np.median(errors)),
                   'xyz_min':xyz.min(0).tolist(),'xyz_max':xyz.max(0).tolist(),'xyz':xyz.tolist()}
    # This is the previously selected legacy track set, so no unbiased calibration GT claim.
    write(REPORT/'camera_audit.json',out)
    print({name:{k:v for k,v in value.items() if k!='xyz'} for name,value in out.items()},flush=True)


if __name__=='__main__':main()
