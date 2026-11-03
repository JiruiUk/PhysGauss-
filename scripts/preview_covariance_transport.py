"""Immutable, hashed preview snapshot of a live I8 run; no model selection."""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw
import torch

from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, write, sha
from variants.staged_reconstruction.covariance_transport.prepare import EXP, WORK


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    from variants.staged_reconstruction.covariance_transport.freeze import verify
    from variants.staged_reconstruction.covariance_transport.train import load_initial
    from variants.staged_reconstruction.covariance_transport.runtime import install, renderer_for
    from variants.staged_reconstruction.train_static import camera, PIPE
    verify()
    live = WORK/'formal/transport/checkpoint.pt'
    before = read(live.with_suffix('.json'))
    step = before['iteration']
    directory = WORK/f'previews/step{step:05d}'
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory/'checkpoint.pt'
    shutil.copyfile(live, snapshot)
    digest = sha(snapshot)
    after = read(live.with_suffix('.json'))
    assert before == after and digest == before['sha256'], 'Concurrent checkpoint changed; preserve this attempt and retry a fresh snapshot'
    state = torch.load(snapshot, map_location='cpu', weights_only=False)
    assert state['iteration'] == step and state['protocol_sha256'] == sha(EXP/'protocol.json')
    renderer, scene, opt, initial, _ = load_initial()
    del opt, initial
    install(scene, read(EXP/'protocol.json'), True)
    scene.load_state_dict(state['model']); del state
    renderer = renderer_for(renderer)
    meta = read(DATA/'transforms_train.json')
    poses = {int(Path(frame['file_path']).stem): frame for frame in meta['frames']}
    out = ROOT/f'output/staged_reconstruction/covariance_transport/previews/step{step:05d}'
    out.mkdir(parents=True, exist_ok=False)
    contact = Image.new('RGB', (800, 3*475), '#171a22')
    draw = ImageDraw.Draw(contact)
    for k, (cam_id, frame) in enumerate([(188, 43), (188, 146), (86, 86)]):
        cam = camera(poses[cam_id], meta, 1600)
        t = (frame-1)/199
        rgb = scene.render(renderer, cam, t)['render'].clamp(0, 1)
        rgb = Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8))
        rgb.save(out/f'c{cam_id}_t{frame}_native.png')
        alpha = renderer(cam, scene.dynamic_at(t), PIPE, scene.centers.new_zeros(3), 0., 0., 0.)['alpha']
        Image.fromarray((alpha.squeeze().clamp(0, 1).cpu().numpy()*255).round().astype(np.uint8)).save(out/f'c{cam_id}_t{frame}_alpha.png')
        contact.paste(rgb.resize((800, 450), Image.Resampling.LANCZOS), (0, k*475+25))
        draw.text((5, k*475+6), f'I8 intermediate preview: {step} updates | c{cam_id}/t{frame} | NOT final acceptance', fill='white')
    contact.save(out/'contact.png')
    assert sha(snapshot) == digest
    record = {'actual_updates': step, 'snapshot': snapshot.relative_to(ROOT).as_posix(), 'snapshot_sha256': digest,
              'source_sha256': sha(Path(__file__)), 'native_width': 1600, 'RGB_read': False, 'optimizer_updates': 0,
              'purpose': 'Intermediate visual diagnosis only; original16k schedule/gates remain. No model selection or final acceptance.',
              'images_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(out.glob('*.png'))}}
    write(out/'receipt.json', record)
    write(EXP/f'preview_{step:05d}.json', record)
    print('COVARIANCE_TRANSPORT_PREVIEW', step, digest, out, flush=True)


if __name__ == '__main__':
    main()
