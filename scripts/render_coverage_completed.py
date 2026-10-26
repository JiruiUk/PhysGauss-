"""User-requested priority C1/C2 final50k export; resume the unchanged C3 queue.

Completed arms have their own explicit early-test lock. The full queue lock is
not fabricated or changed. C3 settings remain fixed despite early test exposure.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from variants.coverage_tuning import run as study
from variants.metric_tuning import run as engine
from variants.kinematic_resplit import run as base

NAMES = ['C1_mask','C2_mask_affine']
PRIORITY = study.RUNS/'priority_render'
RECEIPT = study.REPORTS/'priority_render.json'


def pairs(directory):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image,ImageDraw,ImageFont
    raw = directory/'test/ours_50000'
    gallery = study.RUNS/'render_comparisons'/directory.parent.name
    target = gallery/'pairs/50000'; target.mkdir(parents=True,exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    def pair(i):
        name = f'{i:05d}.png'
        with Image.open(raw/'gt'/name) as a,Image.open(raw/'renders'/name) as b:
            a,b = a.convert('RGB'),b.convert('RGB')
            if a.size!=b.size or a.size!=(1600,900): raise ValueError('Unexpected render resolution')
            image = Image.new('RGB',(3200,950),'#edf2f7'); draw = ImageDraw.Draw(image)
            draw.text((15,10),'GT | '+name,font=font,fill='#17304a')
            draw.text((1615,10),directory.parent.name+' | final50000',font=font,fill='#17304a')
            image.paste(a,(0,50)); image.paste(b,(1600,50))
            path = target/name; image.save(path,compress_level=3)
            with Image.open(path) as check:
                if check.crop((0,50,1600,950)).tobytes()!=a.tobytes() or check.crop((1600,50,3200,950)).tobytes()!=b.tobytes():
                    raise ValueError('GT/render pair pixel verification failed')
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(pair,range(50)))
    page = ('<!doctype html><meta charset="utf-8"><title>GT / '+directory.parent.name+'</title>'
        '<style>body{background:#101725;color:white;font:16px system-ui;margin:25px}img{width:100%}a{color:#86e1bb}</style>'
        '<h1>'+directory.parent.name+' | final50k</h1><p>Left: GT; right: render. 50 official test frames, original frame IDs 1,5,...,197.</p>'
        '<input id="frame" type="range" min="0" max="49" value="25"><span id="number"></span> <a id="full">Open full-size pair</a><img id="image">'
        '<script>const f=document.getElementById("frame");function update(){let i=Number(f.value),n=String(i).padStart(5,"0"),p="pairs/50000/"+n+".png";'
        'document.getElementById("image").src=p;document.getElementById("full").href=p;document.getElementById("number").textContent="Original frame "+(4*i+1)+" | "+n}f.oninput=update;update()</script>')
    engine.write_text_atomic(gallery/'index.html',page)
    manifest = {'run_absolute':str(directory),'gallery_absolute':str(gallery/'index.html'),
        'pairs_absolute':str(target),'pair_count':50,'tags':['50000'],'pixel_identity_verified':True,
        'protocol':'User-requested early export of completed final50k; separate completed-arm test lock'}
    engine.write_json(gallery/'manifest.json',manifest)
    return manifest


def main():
    PRIORITY.mkdir(parents=True,exist_ok=True)
    original = engine.read_json(study.RUNS/'controller.json')
    deadline = datetime.fromisoformat(original['deadline_utc'])
    receipt = {'status':'waiting_for_safe_pause','pid':os.getpid(),'started_utc':engine.now(),
        'user_request':'render completed C1 and C2 first','names':NAMES,'tags':['50000'],
        'original_controller':original,'completed':[],
        'protocol_change':'Completed-arm test exports precede C3 completion at user request; C3 config remains unchanged.'}
    engine.write_json(RECEIPT,receipt)
    wait_until = time.monotonic()+180
    while engine.alive(original['pid']):
        if time.monotonic()>wait_until: raise RuntimeError('Original queue has not safely stopped; no concurrent GPU work launched')
        time.sleep(1)
    paused = engine.read_json(study.RUNS/'controller.json')
    if paused['status']!='saved_stopped': raise ValueError('Original queue did not save and stop')
    c3 = study.RUNS/'C3_mask_affine_density/rubber_duck/checkpoints/resume.pt'
    manifest = engine.read_json(c3.with_suffix('.json'))
    if engine.sha256(c3)!=manifest['sha256']: raise ValueError('C3 full checkpoint checksum mismatch')
    receipt['paused_c3'] = manifest
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    pause = study.RUNS/'stop_queue.request'
    if pause.exists():
        archived = PRIORITY/('pause_request_'+stamp+'.txt'); pause.replace(archived)
        receipt['archived_pause_request'] = str(archived)
    engine.write_json(PRIORITY/'controller_before_priority.json',paused)
    repo = study.prepare(); study.assert_frozen(repo); study.wire(repo)
    # Own lock protects only C1/C2's completed, frozen choices. Main coordinator's
    # eventual all-arm lock remains absent until C3 finishes.
    locked = {}
    for name in NAMES:
        d = study.RUNS/name/'rubber_duck'
        if engine.read_json(d/'checkpoints/resume.json')['iteration']!=50000: raise ValueError('Incomplete requested arm')
        files = [p for folder in ('point_cloud','deform') for p in (d/folder/'iteration_50000').rglob('*') if p.is_file()]
        locked[name] = {'validation_selections':engine.read_json(d/'metric_best.json'),
            'final_checkpoint_files':{str(p):engine.sha256(p) for p in files}}
    lock = PRIORITY/'selection_lock.json'
    if lock.exists():
        if engine.read_json(lock)['locked_models']!=locked: raise ValueError('Previously locked priority snapshots changed')
    else: engine.write_json(lock,{'locked_utc':engine.now(),'locked_models':locked,'tags':['50000'],
        'reason':'Explicit user request: render completed C1 C2 before C3','test_used_for_selection':False})
    receipt.update(status='rendering',selection_lock=str(lock)); engine.write_json(RECEIPT,receipt)
    os.environ['PIDG_EXPORT_ALPHA'] = '1'
    succeeded = False
    try:
        with engine.lock():
            state = dict(paused,pid=os.getpid(),status='priority_render',priority_render_pid=os.getpid())
            engine.write_json(study.RUNS/'controller.json',state); study.report()
            for name in NAMES:
                d = study.RUNS/name/'rubber_duck'
                previous = base.RUNS
                try:
                    base.RUNS = PRIORITY
                    base.render(d,repo,deadline,tags=['50000'])
                finally: base.RUNS = previous
                # Render is read-only with respect to every selected weight.
                for p,digest in locked[name]['final_checkpoint_files'].items():
                    if engine.sha256(Path(p))!=digest: raise ValueError('Render changed frozen weights')
                gallery = pairs(d)
                coverage = study.coverage_scores(d,tags=['50000'])
                receipt['completed'].append({'id':name,**gallery,'coverage':coverage['50000']['mean']})
                engine.write_json(RECEIPT,receipt); study.report()
            receipt.update(status='completed',finished_utc=engine.now()); engine.write_json(RECEIPT,receipt)
            succeeded = True
    except Exception as error:
        receipt.update(status='failed',error=str(error),finished_utc=engine.now()); engine.write_json(RECEIPT,receipt)
        raise
    finally:
        # A new user pause request takes precedence over automatic continuation.
        if not (study.RUNS/'stop_queue.request').exists():
            remaining = (deadline-datetime.now(timezone.utc)).total_seconds()/3600
            if remaining>0:
                out = (study.REPORTS/('resume_after_priority_'+stamp+'.stdout.log')).open('a')
                err = (study.REPORTS/('resume_after_priority_'+stamp+'.stderr.log')).open('a')
                try:
                    child = subprocess.Popen([sys.executable,'-u','-m','variants.coverage_tuning.run',
                        '--hours',str(remaining)],cwd=ROOT,stdout=out,stderr=err,stdin=subprocess.DEVNULL,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                finally: out.close(); err.close()
                receipt['resumed_controller_pid'] = child.pid
                receipt['resume_original_deadline_utc'] = deadline.isoformat()
                engine.write_json(RECEIPT,receipt)
        print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__': main()
