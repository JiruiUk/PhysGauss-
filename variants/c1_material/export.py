"""Collect final50k renders/pairs and a GT/P0/P1/P2 four-panel gallery."""
from concurrent.futures import ThreadPoolExecutor
import json
from scripts import export_color_repair_images as collect
from scripts import export_color_repair_four_panel as four
from .run import RUNS,NAMES,ROOT,REPORTS


def export():
    output=RUNS/'render_exports/final50000';output.mkdir(parents=True,exist_ok=True)
    collect.RUNS,collect.OUTPUT=RUNS,output
    frames=collect.read(collect.SPLIT)['frames']
    manifests=[collect.export_arm(name,frames) for name in NAMES]
    four.SOURCE,four.OUTPUT,four.ARMS=output,output/'gt_p0_p1_p2',tuple(NAMES)
    four.OUTPUT.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(lambda i:four.export_frame(i,[m['frames'][i] for m in manifests]),range(50)))
    record={'status':'completed','images':50,'split':'test','snapshot':50000,'panel_order':['GT',*NAMES],
        'output_absolute':str(four.OUTPUT),'canvas_size':[6400,950],'all_source_pixels_preserved':True,'frames':rows}
    collect.write(four.OUTPUT/'manifest.json',record)
    ids=[r['original_frame_id'] for r in rows]
    page='''<!doctype html><meta charset="utf-8"><title>GT / C1 / fixed E / learned E</title>
<style>body{font:16px system-ui;background:#111c2c;color:#edf5ff;margin:24px}img{width:100%}a{color:#8de4b1}input{width:45%}</style>
<h1>GT | C1 control | fixed E | learned E</h1><p>All models final50000; fixed50 test frames. Each panel1600x900, full PNG6400x950.</p>
<input id="frame" type="range" min="0" max="49" value="0"><span id="number"></span> <a id="full">Open PNG</a><img id="image">
<script>const ids=IDS,f=document.getElementById('frame');function update(){const i=Number(f.value),p=String(i).padStart(5,'0')+'.png';document.getElementById('image').src=p;document.getElementById('full').href=p;document.getElementById('number').textContent='Original frame '+ids[i]+' | '+(i+1)+'/50';}f.oninput=update;update();</script>'''
    collect.atomic_text(four.OUTPUT/'index.html',page.replace('IDS',json.dumps(ids)))
    collect.atomic_text(output/'index.html','<!doctype html><meta charset="utf-8"><h1>C1 material comparison</h1><a href="gt_p0_p1_p2/index.html">GT + all three models</a>'+''.join(
        f'<p>{name}: <a href="{name}/renders/">renders</a> | <a href="{name}/gt_render/">GT/render pairs</a></p>' for name in NAMES))
    collect.write(REPORTS/'render_exports.json',{'status':'completed','models':NAMES,'renders':150,'pairs':150,
        'four_panel_images':50,'gallery_absolute':str(four.OUTPUT/'index.html')})
