"""Locked comparison of repaired variants against unchanged H3/C1 exports."""
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from scripts import export_color_repair_images as collect
from variants.integrity_tuning.diagnose import evaluate_exports
from .run import RUNS,REPORTS,NAMES,REFERENCE,engine,prior


def export_and_assess():
    lock=engine.read_json(RUNS/'selection_lock.json')
    if lock['selection_split']!='validation':raise ValueError('Validation lock required')
    sources={'C1_reused':prior.REFERENCE,'H3_reused':REFERENCE,**{n:RUNS/n/'rubber_duck' for n in NAMES}}
    frames=collect.read(collect.SPLIT)['frames'];output=RUNS/'render_exports/final50000';records=[]
    for name,d in sources.items():
        metrics=engine.read_json(d/'results.json')['ours_50000'];geometry=evaluate_exports(d,'50000',split='test')
        per=engine.read_json(d/'per_view.json')['ours_50000']
        records.append({'model':name,'run_absolute':str(d),'metrics':metrics,'integrity':geometry,'per_view':per,
            'worst5_PSNR':sum(sorted(per['PSNR'].values())[:5])/5})
        with ThreadPoolExecutor(max_workers=3) as pool:
            copied=list(pool.map(lambda item:collect.export_frame(d/'test/ours_50000',output,name,*item),enumerate(frames)))
        collect.write(output/name/'manifest.json',{'frames':copied,'source':str(d),'snapshot':50000})
    folder=output/'gt_h3_repairs';folder.mkdir(exist_ok=True)
    def join(i):
        filename=f'{i:05d}.png';frame=int(Path(frames[i]['file_path']).stem)
        panels=[('GT',output/'gt'/filename)]+[(n,output/n/'renders'/filename) for n in sources]
        image=Image.new('RGB',(len(panels)*1600,950),'#edf2f7');draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
        hashes=[]
        for j,(name,path) in enumerate(panels):
            with Image.open(path) as im:image.paste(im.convert('RGB'),(j*1600,50));hashes.append(im.convert('RGB').tobytes())
            draw.text((j*1600+12,10),f'{name} | frame {frame:04d} | final50000',font=font,fill='#17304a')
        destination=folder/filename;image.save(destination,compress_level=3)
        with Image.open(destination) as im:
            assert all(im.crop((j*1600,50,(j+1)*1600,950)).tobytes()==h for j,h in enumerate(hashes))
        return {'index':i,'frame':frame,'sha256':collect.digest(destination)}
    with ThreadPoolExecutor(max_workers=3) as pool:joined=list(pool.map(join,range(50)))
    collect.write(folder/'manifest.json',{'frames':joined,'source_pixels_preserved':True,'panel_order':['GT',*sources]})
    collect.atomic_text(folder/'index.html','<!doctype html><meta charset="utf-8"><title>GT / H3 / repairs</title><style>body{font:16px system-ui;margin:24px;background:#142033;color:white}img{width:100%}a{color:#8de4b1}</style><h1>GT | C1 | H3 | D1 | D2 | D3</h1><input id="f" type="range" min="0" max="49"><a id="a">Open original-size PNG</a><img id="i"><script>function update(){let p=String(+f.value).padStart(5,"0")+".png";i.src=p;a.href=p}f.oninput=update;update()</script>')
    engine.write_json(REPORTS/'assessment.json',{'selected_by_validation':lock['selected'],'models':records,'gallery_absolute':str(folder/'index.html'),
        'test_exposure':'Historical test exposure persists. No test targets entered repair training or selection.',
        'completeness_scope':'2D alpha and appearance diagnostics; not proof of3D watertightness'})
    with (REPORTS/'per_frame_test.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame_id',*[r['model'] for r in records]])
        for i,f in enumerate(frames):writer.writerow([int(Path(f['file_path']).stem),*[r['per_view']['PSNR'][f'{i:05d}.png'] for r in records]])
    lines=['# H3 worst-frame repair: final50k comparison','',
        'Model chosen on validation: '+lock['selected']+'. No improvement is assumed from adding a module.',
        '','| Model | PSNR | SSIM | LPIPS | MS-SSIM | Worst5 PSNR | Core low90 % | BG spill % |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['metrics'];g=r['integrity']['mean'];lines.append(f"| {r['model']} | {m['PSNR']:.6f} | {m['SSIM']:.7f} | {m['LPIPS']:.7f} | {m['MS-SSIM']:.7f} | {r['worst5_PSNR']:.5f} | {100*g['core_low_90']:.6f} | {100*g['background_false_coverage']:.6f} |")
    lines+=['','GT/render gallery: '+str(folder/'index.html'),'','Every source panel preserves1600x900 pixels. Each variant also has renders/ and gt_render/ directories.']
    engine.write_text_atomic(REPORTS/'assessment.md','\n'.join(lines)+'\n')
