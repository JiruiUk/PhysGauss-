"""Read-only quarantine preview and predeclared structure-retention check."""
import numpy as np
from PIL import Image, ImageDraw
import torch
from ..prepare_data import ROOT, DATA, read, write, sha
from ..train_static import setup, camera, render_static, optimizer_for
from ..automatic_static.discover import rgb
from ..adaptive_static.model import AdaptiveGaussian, replace_rows
from ..adaptive_static.prepare import RUNS as PREVIOUS_RUNS
from ..cross_temporal_static.evaluate import pixels, metric
from ..cross_temporal_static.prepare import region_confidence
from .prepare import REPORT, RUNS, OUT


@torch.no_grad()
def main():
    protocol = read(REPORT/'protocol.json'); parent = ROOT/protocol['parent']; assert sha(parent) == protocol['parent_sha256']
    state = torch.load(parent, map_location='cpu', weights_only=False); renderer = setup(); torch.set_num_threads(4)
    with np.load(RUNS/'parent_evidence.npz') as data: quarantine = data['quarantine'].copy()
    models = {}
    for name in ['parent', 'quarantine']:
        model = AdaptiveGaussian(PREVIOUS_RUNS/'initialization.npz'); model.load_payload(state['model'])
        optimizer = optimizer_for(model); optimizer.load_state_dict(state['optimizer'])
        if name == 'quarantine': replace_rows(model, optimizer, ~quarantine)
        models[name] = model
    meta = read(DATA/'transforms_train.json'); panels = []
    for number in [2,27,43,86,166,188,200]:
        frame = next(f for f in meta['frames'] if int(f['file_path'].replace('\\','/').split('/')[-1]) == number)
        panel = Image.new('RGB', (1200,250), '#161922')
        images = [('Observed RGB', (rgb(frame,400)*255).round().astype(np.uint8))]
        for name, model in models.items():
            value = pixels(render_static(renderer, camera(frame,meta,1600), model)['render'])
            Image.fromarray(value).save(OUT/f'probe_{name}_t{number:04d}_native.png'); images.append((name,value))
        for i,(name,value) in enumerate(images):
            panel.paste(Image.fromarray(value).resize((400,225),Image.Resampling.LANCZOS),(i*400,25))
            ImageDraw.Draw(panel).text((i*400+5,5),f't{number} {name}',fill='white')
        panels.append(np.asarray(panel))
    Image.fromarray(np.concatenate(panels)).save(OUT/'quarantine_preview.png')
    old = ROOT/'experiments/staged_reconstruction/S1_cross_temporal'; discovery, old_protocol = read(old/'discovery.json'), read(old/'protocol.json')
    validation = read(ROOT/'data/derived/kinematic_resplit/transforms_val.json'); results = {n:[] for n in models}
    for frame in validation['frames']:
        _, mask = region_confidence(frame, rgb(frame,400), discovery['anchors'], discovery['incompatible'], validation, old_protocol)
        gt = torch.from_numpy(rgb(frame,1600)).permute(2,0,1).cuda(); cam = camera(frame,validation,1600)
        for name,model in models.items(): results[name].append(metric(render_static(renderer,cam,model),gt,mask >= .5))
    aggregate = {n:{key:float(np.mean([v[key] for v in values if v is not None])) for key in ['PSNR','alpha_mean','alpha_below_05']} for n,values in results.items()}
    passed = bool(quarantine.any() and (~quarantine).mean() >= .75 and
        aggregate['quarantine']['PSNR'] >= aggregate['parent']['PSNR']-.5 and
        aggregate['quarantine']['alpha_mean'] >= aggregate['parent']['alpha_mean']-.02)
    write(REPORT/'structure_probe.json', {'passed':passed,'parent_sha256':sha(parent),'quarantined':int(quarantine.sum()),
        'remaining':int((~quarantine).sum()),'frozen_CT_val_inferred_not_GT':aggregate,
        'predeclared_gate':'retain>=75% rows, frozen inferred validation PSNR loss<=0.5dB, alpha_mean loss<=0.02; manual PNG review also required',
        'manual_review_pending':True,'independent_scene_geometry_GT':False})
    print('CLEAN_STATIC_PROBE', passed, aggregate, flush=True)


if __name__ == '__main__': main()
