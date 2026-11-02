"""Read-only visualization of training flow provenance and confidence."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/'data/derived/flow_comparison_train_only'
SOURCE = ROOT/'data/derived/kinematic_resplit/training'
manifest = json.loads((CACHE/'manifest.json').read_text())
frames = json.loads((SOURCE/'transforms_train.json').read_text())['frames']
paths = {int(Path(f['file_path']).stem): SOURCE/(f['file_path']+'.png') for f in frames}
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
canvas = Image.new('RGB', (1200, 3*265), '#edf2f7'); draw = ImageDraw.Draw(canvas)
rows = []
for index, pair in enumerate([manifest['pairs'][0], manifest['pairs'][48], manifest['pairs'][-1]]):
    with Image.open(paths[pair['source']]) as original:
        rgba = np.asarray(original.convert('RGBA'), dtype=np.float32)
    image = Image.fromarray((rgba[..., :3]*rgba[..., 3:4]/255).astype(np.uint8)).resize((400, 225))
    with np.load(CACHE/pair['file']) as data:
        valid = data['valid']; raw = data['forward']; motion = data['motion']
    for column, (label, field) in enumerate([('Observed flow (camera + motion)', raw), ('Camera-compensated motion estimate', motion)]):
        tile = image.copy(); arrows = ImageDraw.Draw(tile)
        for y in range(12, 225, 12):
            for x in range(12, 400, 12):
                if valid[y, x]:
                    dx, dy = field[y, x]
                    arrows.line((x, y, x+float(dx), y+float(dy)), fill='#ff653b', width=2)
                    arrows.ellipse((x-1, y-1, x+1, y+1), fill='#ffef54')
        canvas.paste(tile, (400*column, 265*index+40))
        draw.text((400*column+6, 265*index+5), label, font=font, fill='#17304a')
    confidence = Image.fromarray((valid.astype(np.uint8)*255)).convert('RGB')
    canvas.paste(confidence, (800, 265*index+40))
    draw.text((806, 265*index+5), f"{pair['source']} -> {pair['target']} | valid={valid.sum()}", font=font, fill='#17304a')
    rows.append({'source': pair['source'], 'target': pair['target'], 'dt': pair['dt'], 'valid_pixels': int(valid.sum()),
        'observed_median_px': float(np.median(np.linalg.norm(raw[valid], axis=-1))),
        'motion_median_px': float(np.median(np.linalg.norm(motion[valid], axis=-1)))})
out = Path(__file__).parent
canvas.save(out/'training_flow_inputs.png')
(out/'input_qa.json').write_text(json.dumps({'examples': rows, 'pairs': len(manifest['pairs']),
    'min_valid_pixels': min(p['valid_pixels'] for p in manifest['pairs']),
    'median_valid_pixels': float(np.median([p['valid_pixels'] for p in manifest['pairs']])),
    'max_training_gap_frames': max(p['target']-p['source'] for p in manifest['pairs'])}, indent=2))
