"""Pinned same-step I8/I9 native comparison, no training or acceptance changes."""
import argparse
import itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from variants.staged_reconstruction.prepare_data import ROOT, REPORT, RUNS, DATA, read, write, sha


def sampled_frames(branch, arm, step):
    path = RUNS/branch/'formal'/arm/'steps.jsonl'
    with path.open(encoding='utf-8') as stream:
        rows = [__import__('json').loads(line) for line in itertools.islice(stream, step)]
    assert len(rows) == step and [r['global_step'] for r in rows] == list(range(1, step+1))
    assert all('physics' not in r for r in rows)
    assert all(r['width'] == (800 if r['global_step'] <= 12000 else 1600) for r in rows)
    return [row['frame_index'] for row in rows]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--step', type=int, required=True); args = p.parse_args()
    assert 0 < args.step < 16000
    torch.set_num_threads(4)
    branches = [('I8', 'covariance_transport', 'transport'), ('I9', 'relative_strain', 'strain')]
    receipts, states, initials = {}, {}, {}
    for label, branch, arm in branches:
        receipt = read(REPORT/branch/f'preview_{args.step:05d}.json')
        assert receipt['actual_updates'] == args.step and receipt['native_width'] == 1600
        assert sha(ROOT/receipt['snapshot']) == receipt['snapshot_sha256']
        for image_path, digest in receipt['images_sha256'].items(): assert sha(ROOT/image_path) == digest
        state = torch.load(ROOT/receipt['snapshot'], map_location='cpu', weights_only=False)
        assert state['iteration'] == args.step and state['protocol_sha256'] == sha(REPORT/branch/'protocol.json')
        receipts[label] = receipt; states[label] = state
        initials[label] = torch.load(RUNS/branch/'formal'/arm/'initial.pt', map_location='cpu', weights_only=False)
    a, b = initials['I8'], initials['I9']
    assert a['iteration'] == b['iteration'] == 0 and a['model'].keys() == b['model'].keys()
    assert all(torch.equal(value, b['model'][key]) for key, value in a['model'].items())
    assert a['optimizer'] == b['optimizer']
    assert states['I8']['optimizer']['param_groups'] == states['I9']['optimizer']['param_groups']
    assert states['I8']['camera_stack'] == states['I9']['camera_stack']
    assert states['I8']['rng_python'] == states['I9']['rng_python']
    assert torch.equal(states['I8']['rng_torch'], states['I9']['rng_torch'])
    assert all(torch.equal(x,y) for x,y in zip(states['I8']['rng_cuda'], states['I9']['rng_cuda']))
    for index,(x,y) in enumerate(zip(states['I8']['rng_numpy'],states['I9']['rng_numpy'])):
        assert np.array_equal(x,y), index
    assert sampled_frames('covariance_transport','transport',args.step) == sampled_frames('relative_strain','strain',args.step)
    out = ROOT/f'output/staged_reconstruction/relative_strain/compare_step{args.step:05d}'
    assert not out.exists(); out.mkdir(parents=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    panels, boxes, inputs = [], {}, {}
    for cam, frame in [(188,43),(188,146),(86,86)]:
        key = f'c{cam}_t{frame}'
        masks, pictures = [], []
        for label in ['I8','I9']:
            files = receipts[label]['images_sha256']
            rgb = next(path for path in files if path.endswith(f'/{key}_native.png'))
            alpha = next(path for path in files if path.endswith(f'/{key}_alpha.png'))
            pictures.append(Image.open(ROOT/rgb).convert('RGB'))
            masks.append(np.asarray(Image.open(ROOT/alpha)))
            inputs[f'{label}/{key}'] = {'RGB':rgb,'RGB_sha256':files[rgb],'alpha':alpha,'alpha_sha256':files[alpha]}
        y,x = np.nonzero(np.maximum.reduce(masks)>12)
        box = (max(0,int(x.min())-40),max(0,int(y.min())-40),min(1600,int(x.max())+41),min(900,int(y.max())+41))
        boxes[key] = box
        width = 800; height = round((box[3]-box[1])*width/(box[2]-box[0]))
        panel = Image.new('RGB',(width*2,height+40),'#171a22'); draw=ImageDraw.Draw(panel)
        for column,(label,picture) in enumerate(zip(['I8','I9'],pictures)):
            panel.paste(picture.crop(box).resize((width,height),Image.Resampling.LANCZOS),(column*width,40))
            draw.text((column*width+8,8),f'{label} step{args.step} | c{cam}/t{frame}',font=font,fill='white')
        panel.save(out/f'{key}_comparison.png'); panels.append(panel)
    sheet=Image.new('RGB',(1600,sum(panel.height for panel in panels)),'#171a22')
    cursor=0
    for panel in panels: sheet.paste(panel,(0,cursor)); cursor+=panel.height
    sheet.save(out/'comparison.png')
    write(REPORT/'relative_strain'/f'comparison{args.step:05d}.json',{
        'actual_updates_each':args.step,'script_sha256':sha(__file__),
        'checkpoints':{label:{'path':r['snapshot'],'sha256':r['snapshot_sha256']} for label,r in receipts.items()},
        'extended_initial_model_tensors_exact':True,'initial_optimizer_exact':True,
        'same_optimizer_groups_and_sampler_and_CPU_CUDA_numpy_rng':True,
        'same_original_resolution_schedule':True,'physics_calls_each':0,
        'training_metadata_sha256':sha(DATA/'transforms_train.json'),
        'camera_contract':'Same frozen camera() function, training metadata poses by frame ID, native1600 and (frame-1)/199 in both pinned preview sources',
        'inputs':inputs,'common_crop_per_row':boxes,'output_panel_width_per_model':800,
        'resize':'Common crop per row, Lanczos; unmodified original1600 images preserved',
        'outputs_sha256':{path.relative_to(ROOT).as_posix():sha(path) for path in sorted(out.glob('*.png'))},
        'optimizer_updates_by_diagnostic':0,'test_RGB_read':False,'final_candidate_selected':False,
        'acceptance':'Not evaluated; intermediate comparison only'})
    print('SAME_STEP_TRANSPORT_STRAIN_COMPARISON_SAVED',args.step,out,flush=True)


if __name__=='__main__': main()
