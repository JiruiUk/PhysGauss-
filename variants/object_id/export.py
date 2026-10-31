"""Final50k GT/C1/frozen-ID/learned-ID exports with pixel-preserving panels."""
from concurrent.futures import ThreadPoolExecutor
import json
from scripts import export_color_repair_images as collect
from scripts import export_color_repair_four_panel as four
from .run import RUNS,NAMES,REPORTS,run_directory


def export_arm(name,frames,output):
    source=run_directory(name);raw=source/'test/ours_50000'
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(lambda item:collect.export_frame(raw,output,name,*item),enumerate(frames)))
    manifest={'status':'completed','arm':name,'iteration':50000,'split':'test','images':50,
        'source_run':str(source),'baseline_reused':name==NAMES[0],
        'source_checkpoint_sha256':collect.read(source/'checkpoints/resume.json')['sha256'],
        'source_render_receipt_sha256':collect.digest(raw/'render_complete.json'),
        'split_sha256':collect.digest(collect.SPLIT),'source_image_bytes_preserved':True,
        'pair_pixels_verified':True,'frames':records}
    collect.write(output/name/'manifest.json',manifest)
    return manifest


def export():
    output=RUNS/'render_exports/final50000';output.mkdir(parents=True,exist_ok=True)
    frames=collect.read(collect.SPLIT)['frames']
    manifests=[export_arm(name,frames,output) for name in NAMES]
    four.SOURCE,four.OUTPUT,four.ARMS=output,output/'gt_c1_frozen_learned',tuple(NAMES)
    four.OUTPUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(lambda i:four.export_frame(i,[m['frames'][i] for m in manifests]),range(50)))
    record={'status':'completed','images':50,'split':'test','snapshot':50000,'panel_order':['GT',*NAMES],
        'output_absolute':str(four.OUTPUT),'canvas_size':[6400,950],'all_source_pixels_preserved':True,'frames':rows}
    collect.write(four.OUTPUT/'manifest.json',record)
    ids=[r['original_frame_id'] for r in rows]
    page='''<!doctype html><meta charset="utf-8"><title>GT / C1 / frozen ID / learned ID</title>
<style>body{font:16px system-ui;background:#142033;color:#edf5ff;margin:24px}img{width:100%}a{color:#8de4b1}input{width:45%}</style>
<h1>GT | C1 | frozen identity | learned identity</h1><p>All models total50000; same50 final-test frames. No instance labels used for identity training.</p>
<input id="frame" type="range" min="0" max="49" value="0"><span id="number"></span> <a id="full">Open PNG</a><img id="image">
<script>const ids=IDS,f=document.getElementById('frame');function update(){const i=Number(f.value),p=String(i).padStart(5,'0')+'.png';document.getElementById('image').src=p;document.getElementById('full').href=p;document.getElementById('number').textContent='Original frame '+ids[i]+' | '+(i+1)+'/50';}f.oninput=update;update();</script>'''
    collect.atomic_text(four.OUTPUT/'index.html',page.replace('IDS',json.dumps(ids)))
    collect.atomic_text(output/'index.html','<!doctype html><meta charset="utf-8"><h1>C1 Object-ID comparison</h1><a href="gt_c1_frozen_learned/index.html">GT + all three models</a>'+''.join(
        f'<p>{name}: <a href="{name}/renders/">renders</a> | <a href="{name}/gt_render/">GT/render pairs</a></p>' for name in NAMES))
    collect.write(REPORTS/'render_exports.json',{'status':'completed','models':NAMES,'renders':150,'pairs':150,
        'four_panel_images':50,'gallery_absolute':str(four.OUTPUT/'index.html')})
