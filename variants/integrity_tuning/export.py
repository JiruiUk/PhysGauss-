"""Locked final-test assessment and GT/baseline/promoted-model image exports."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from scripts import export_color_repair_images as collect
from .run import RUNS,REPORTS,REFERENCE,base,engine,read_history,quality_guard
from .diagnose import evaluate_exports


def export_and_assess(names):
    if not (RUNS/'selection_lock.json').exists():raise ValueError('Integrity test is still locked')
    baseline=engine.read_json(REPORTS/'diagnosis.json')
    baseline_test=evaluate_exports(REFERENCE,'50000',split='test')
    baseline_test_metrics=engine.read_json(REFERENCE/'results.json')['ours_50000']
    records=[]
    for name in names:
        d=RUNS/name/'rubber_duck';validation=next(r for r in reversed(read_history(d)) if r['iteration']==50000)
        test=evaluate_exports(d,'50000',split='test');metrics=engine.read_json(d/'results.json')['ours_50000']
        improved=quality_guard(validation['metrics'],baseline['image_metrics']) and \
            validation['integrity']['integrity_score']<baseline['mean']['integrity_score']-1e-5 and \
            validation['integrity']['background_false_coverage']<=baseline['mean']['background_false_coverage']+.002
        records.append({'model':name,'validation':validation,'test':test,'test_metrics':metrics,
            'validation_supported_improvement':improved,'test_metric_delta':{k:metrics[k]-baseline_test_metrics[k] for k in engine.METRICS},
            'integrity_best':engine.read_json(d/'integrity_best.json') if (d/'integrity_best.json').exists() else None})
    output=RUNS/'render_exports/final50000';output.mkdir(parents=True,exist_ok=True)
    sources={'C1_reused':REFERENCE,**{n:RUNS/n/'rubber_duck' for n in names}}
    frames=collect.read(collect.SPLIT)['frames'];exports={}
    for name,directory in sources.items():
        raw=directory/'test/ours_50000'
        with ThreadPoolExecutor(max_workers=3) as pool:
            rows=list(pool.map(lambda item:collect.export_frame(raw,output,name,*item),enumerate(frames)))
        exports[name]={'source':str(directory),'baseline_reused':name=='C1_reused','frames':rows}
        collect.write(output/name/'manifest.json',exports[name])
    folder=output/'gt_baseline_variants';folder.mkdir(exist_ok=True)
    def combine(i):
        filename=f'{i:05d}.png';original_id=int(Path(frames[i]['file_path']).name)
        panels=[('GT',output/'gt'/filename)]+[(n,output/n/'renders'/filename) for n in sources]
        canvas=Image.new('RGB',(1600*len(panels),950),'#edf2f7');draw=ImageDraw.Draw(canvas)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25);checks=[]
        for j,(name,path) in enumerate(panels):
            with Image.open(path) as im:
                im=im.convert('RGB');canvas.paste(im,(j*1600,50));checks.append(im.tobytes())
            draw.text((j*1600+12,10),f'{name} / frame{original_id:04d} / final50000',font=font,fill='#17304a')
        target=folder/filename;canvas.save(target,compress_level=3)
        with Image.open(target) as im:
            for j,pixels in enumerate(checks):
                if im.crop((j*1600,50,(j+1)*1600,950)).tobytes()!=pixels:raise ValueError('Integrity concatenation changed pixels')
        return {'index':i,'frame':original_id,'sha256':collect.digest(target)}
    with ThreadPoolExecutor(max_workers=3) as pool:joined=list(pool.map(combine,range(50)))
    collect.write(folder/'manifest.json',{'split':'test','images':50,'panel_order':['GT',*sources],'source_pixels_preserved':True,'frames':joined})
    page='''<!doctype html><meta charset="utf-8"><title>Object completeness comparison</title><style>body{font:16px system-ui;background:#142033;color:#edf5ff;margin:24px}img{width:100%}a{color:#8de4b1}</style><h1>GT | reused C1 | promoted variants</h1><p>Same50 final-test frames; validation-selected experiments. All panels final50000.</p><input id="f" type="range" min="0" max="49" value="0"><a id="a">Open PNG</a><img id="i"><script>function u(){let p=String(+f.value).padStart(5,'0')+'.png';i.src=p;a.href=p}f.oninput=u;u();</script>'''
    collect.atomic_text(folder/'index.html',page)
    collect.atomic_text(output/'index.html','<!doctype html><meta charset="utf-8"><h1>Completeness experiment renders</h1><a href="gt_baseline_variants/index.html">GT and all models</a>'+''.join(
        f'<p>{n}: <a href="{n}/renders/">render images</a> | <a href="{n}/gt_render/">GT/render pairs</a></p>' for n in sources))
    result={'status':'completed','baseline_reused':str(REFERENCE),'baseline_validation':baseline['mean'],
        'baseline_test':baseline_test,'baseline_test_metrics':baseline_test_metrics,'models':records,
        'recommendation_uses':'validation only; final test reports do not change promotion or model selection',
        'limitations':['single scene/seed','historical test exposure','2D silhouette proxies do not prove3D watertightness',
            'persistent/color neighbor gates do not provide true object identities'],
        'gallery_absolute':str(folder/'index.html')}
    engine.write_json(REPORTS/'assessment.json',result)
    lines=['# Completeness-first experiment assessment','',
        'C1 is reused; all promoted models trained to50k total from the identical full3k state.',
        'Promotion and checkpoint selection used validation only. Test data did not select variants.',
        'Success requires a lower validation integrity score, four-metric quality guard, and background spill within+0.2 percentage point.','',
        '| Model | Validation-supported improvement | Test PSNR | SSIM | LPIPS | MS-SSIM | Test core low50 | Core low90 | BG spill |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['test_metrics'];g=r['test']['mean']
        lines.append('| '+r['model']+' | '+str(r['validation_supported_improvement'])+' | '+' | '.join(f'{m[k]:.7f}' for k in engine.METRICS)+' | '+' | '.join(f'{g[k]:.7f}' for k in ('core_low_50','core_low_90','background_false_coverage'))+' |')
    lines+=['','These are2D opacity/appearance and projected-connectivity diagnostics, not proof of complete3D surfaces.',
        'Opaque appearance errors are reported separately from holes. No learned semantic/material segmentation was added.',
        '','GT + baseline + promoted models: '+str(folder/'index.html')]
    engine.write_text_atomic(REPORTS/'assessment.md','\n'.join(lines)+'\n')
