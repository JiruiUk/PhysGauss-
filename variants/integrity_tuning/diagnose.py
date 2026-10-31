"""Read existing validation exports; never opens final-test RGB for tuning."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base
from variants.coverage_tuning.data import read_alpha
from .model import image_metrics,aggregate
ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'experiments/integrity_tuning'
PREVIEW=ROOT/'runs/pidg_color_repair/validation_previews/C1_repaired_50000'


def evaluate_exports(directory,tag,split='validation'):
    source=base.DATA/('training' if split=='validation' else 'final_test')
    frames=engine.read_json(source/'transforms_test.json')['frames'];rows=[]
    raw=directory/'test'/('ours_'+tag)
    for i,frame in enumerate(frames):
        name=f'{i:05d}.png'
        with Image.open(raw/'alpha'/name) as im:alpha=np.asarray(im.convert('L'),dtype=np.float32)/255.;size=im.size
        target=read_alpha(source/(frame['file_path']+'.png'),size).numpy()
        with Image.open(raw/'renders'/name) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
        with Image.open(raw/'gt'/name) as im:gt=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
        rows.append({'index':i,'frame':Path(frame['file_path']).stem,**image_metrics(alpha,target,rgb,gt)})
    return {'split':split,'count':len(rows),'snapshot':tag,'mean':aggregate(rows),'per_frame':rows,
        'definition':'2D alpha/appearance proxies; GT connected regions are not semantic instances or 3D watertightness'}


def diagnose():
    REPORTS.mkdir(parents=True,exist_ok=True)
    manifest=engine.read_json(PREVIEW/'manifest.json')
    if manifest['split']!='validation' or manifest['images']!=30 or not manifest['validation_png_parity']:
        raise ValueError('Need existing verified validation30 export')
    result=evaluate_exports(PREVIEW,'50000')
    result.update(image_metrics=manifest['metrics'],original_export=str(PREVIEW),final_test_opened=False)
    worst=sorted(result['per_frame'],key=lambda r:r['integrity_score'],reverse=True)[:5]
    folder=REPORTS/'diagnosis_images';folder.mkdir(exist_ok=True)
    metadata=engine.read_json(base.DATA/'training/transforms_test.json')['frames']
    for row in worst:
        name=f"{row['index']:05d}.png";raw=PREVIEW/'test/ours_50000'
        with Image.open(raw/'alpha'/name) as im:a=np.asarray(im.convert('L'),dtype=np.float32)/255.;size=im.size
        target=read_alpha(base.DATA/'training'/(metadata[row['index']]['file_path']+'.png'),size).numpy()[0]
        overlay=np.zeros((*a.shape,3),dtype=np.uint8)
        overlay[(target>.95)&(a<.9)]=[255,80,50]
        overlay[(target<=.5)&(a>.5)]=[60,150,255]
        panels=[Image.open(raw/'gt'/name).convert('RGB'),Image.open(raw/'renders'/name).convert('RGB'),Image.fromarray((a*255).astype('uint8')).convert('RGB'),Image.fromarray(overlay)]
        canvas=Image.new('RGB',(3200,500),'#edf2f7');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
        for col,(title,im) in enumerate(zip(['GT','C1 RGB','C1 alpha','Red: low cover; blue: spill'],panels)):
            canvas.paste(im.resize((800,450)),(col*800,50));draw.text((col*800+12,10),title,font=font,fill='#17304a')
        canvas.save(folder/f"frame_{row['frame']}.png")
        for im in panels:im.close()
    result['worst_frames']=worst
    engine.write_json(REPORTS/'diagnosis.json',result)
    lines=['# C1 object-completeness diagnosis','',
        'Existing30 validation images only; no baseline training or GPU rendering repeated.',
        'Alpha holes, disconnected image regions and opaque appearance errors are measured separately.',
        'No semantic segmentation target or ground-truth3D surface is introduced.','',
        '| Metric (fractions, lower is better except IoU) | Value |','|---|---:|']
    lines += [f'| {k} | {v:.8f} |' for k,v in result['mean'].items()]
    lines += ['','Worst validation frames: '+', '.join(r['frame'] for r in worst),
        '','Foreground/GT-component metrics are2D proxies, not proof of watertight3D objects.',
        'Appearance errors at high alpha must not be reported as opacity holes.',
        '','Diagnostics: '+str(folder),'','Baseline: '+str(PREVIEW)]
    engine.write_text_atomic(REPORTS/'diagnosis.md','\n'.join(lines)+'\n')
    return result


if __name__=='__main__':diagnose()
