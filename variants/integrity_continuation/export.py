"""Full50k comparison; reuse completed reference renders and preserve image pixels."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from scripts import export_color_repair_images as collect
from variants.integrity_tuning.diagnose import evaluate_exports
from .run import JOBS,REFERENCES,RUNS,REPORTS,engine,prior


def export_and_assess():
    if not (RUNS/'selection_lock.json').exists():raise ValueError('Continuation final-test selection is not locked')
    sources={**REFERENCES,**{j['id']:j['run'] for j in JOBS}}
    baseline_validation=engine.read_json(prior.REPORTS/'diagnosis.json')
    records=[]
    for name,d in sources.items():
        metrics=engine.read_json(d/'results.json')['ours_50000'];geometry=evaluate_exports(d,'50000',split='test')
        if name=='C1_reused':validation={'metrics':baseline_validation['image_metrics'],'integrity':baseline_validation['mean']}
        else:validation=next(r for r in reversed(prior.read_history(d)) if r['iteration']==50000)
        record={'model':name,'run_absolute':str(d),'test_metrics':metrics,'test_integrity':geometry,
            'validation':validation,'continued_in_this_round':name not in REFERENCES}
        records.append(record)
    baseline=records[0]['test_metrics']
    for r in records:
        r['test_metric_delta_vs_C1']={k:r['test_metrics'][k]-baseline[k] for k in engine.METRICS}
    output=RUNS/'render_exports/final50000';output.mkdir(parents=True,exist_ok=True)
    frames=collect.read(collect.SPLIT)['frames']
    for name,d in sources.items():
        with ThreadPoolExecutor(max_workers=3) as pool:
            exports=list(pool.map(lambda item:collect.export_frame(d/'test/ours_50000',output,name,*item),enumerate(frames)))
        collect.write(output/name/'manifest.json',{'source':str(d),'snapshot':50000,'split':'test','images':50,'reused':name in REFERENCES,'frames':exports})
    folder=output/'gt_all_models';folder.mkdir(exist_ok=True)
    def concatenate(index):
        file=f'{index:05d}.png';frame=int(Path(frames[index]['file_path']).name)
        panels=[('GT',output/'gt'/file)]+[(n,output/n/'renders'/file) for n in sources]
        image=Image.new('RGB',(1600*len(panels),950),'#edf2f7');draw=ImageDraw.Draw(image)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24);pixels=[]
        for col,(name,path) in enumerate(panels):
            with Image.open(path) as im:
                im=im.convert('RGB');image.paste(im,(col*1600,50));pixels.append(im.tobytes())
            draw.text((col*1600+12,10),f'{name} | frame{frame:04d} | final50000',font=font,fill='#17304a')
        target=folder/file;image.save(target,compress_level=3)
        with Image.open(target) as im:
            if any(im.crop((i*1600,50,(i+1)*1600,950)).tobytes()!=p for i,p in enumerate(pixels)):
                raise ValueError('Continuation comparison altered source pixels')
        return {'index':index,'original_frame_id':frame,'filename':file,'sha256':collect.digest(target)}
    with ThreadPoolExecutor(max_workers=3) as pool:joined=list(pool.map(concatenate,range(50)))
    collect.write(folder/'manifest.json',{'images':50,'panel_order':['GT',*sources],'size':[1600*(len(sources)+1),950],'source_pixels_preserved':True,'frames':joined})
    page='''<!doctype html><meta charset="utf-8"><title>GT and all50k models</title><style>body{font:16px system-ui;background:#142033;color:#edf5ff;margin:24px}img{width:100%}a{color:#8de4b1}</style><h1>GT | C1 | H2 | H3 | H1 | H4 | F1 | F2</h1><p>All models total50000; C1/H2/H3 reused. Original1600x900 panels preserved.</p><input id="f" type="range" min="0" max="49" value="0"><a id="a">Open full-resolution PNG</a><img id="i"><script>function update(){let p=String(+f.value).padStart(5,'0')+'.png';i.src=p;a.href=p}f.oninput=update;update();</script>'''
    collect.atomic_text(folder/'index.html',page)
    collect.atomic_text(output/'index.html','<!doctype html><meta charset="utf-8"><h1>Completed50k model renders</h1><a href="gt_all_models/index.html">GT and all seven models</a>'+''.join(
        f'<p>{n}: <a href="{n}/renders/">render images</a> | <a href="{n}/gt_render/">GT pairs</a></p>' for n in sources))
    result={'status':'completed','snapshot':50000,'split':'fixed50 final-test frames','models':records,
        'historical_test_exposure':True,'user_requested_extension_of_rejected_pilots':True,
        'prior_screening_decisions_preserved':True,'gallery_absolute':str(folder/'index.html'),
        'limitations':['single scene/seed','2D completeness proxies do not establish3D watertightness']}
    engine.write_json(REPORTS/'assessment.json',result)
    lines=['# Continued variants: full50k comparison','',
        'H1/H4/F1/F2 resumed original20k full checkpoints to50k total. C1/H2/H3 reused without training.',
        'Original recipes, validation selection and optimizer/RNG state were retained. Earlier pilot rejection remains recorded.',
        '','| Model | Test PSNR | Delta vs C1 | SSIM | LPIPS | MS-SSIM | Interior low50 (%) | Interior low90 (%) | BG spill (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['test_metrics'];g=r['test_integrity']['mean']
        lines.append(f"| {r['model']} | {m['PSNR']:.7f} | {r['test_metric_delta_vs_C1']['PSNR']:+.7f} | {m['SSIM']:.7f} | {m['LPIPS']:.7f} | {m['MS-SSIM']:.7f} | {100*g['core_low_50']:.7f} | {100*g['core_low_90']:.7f} | {100*g['background_false_coverage']:.7f} |")
    lines+=['','These ratios describe2D rendered alpha/appearance, not3D holes. No improvement is assumed merely because a run reached50k.',
        '','GT/render gallery: '+str(folder/'index.html')]
    engine.write_text_atomic(REPORTS/'assessment.md','\n'.join(lines)+'\n')
