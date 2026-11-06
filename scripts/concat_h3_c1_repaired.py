"""Pair existing final50k test renders; no model loading or GPU inference."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/pidg_integrity_tuning/render_exports/final50000'
OUT=ROOT/'output/comparisons/H3_vs_C1_repaired_test50_50k'
SIZE=(3200,960)
HEADER=60
FPS=5
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--include-s6',action='store_true')
    args=parser.parse_args()
    OUT=ROOT/'output/comparisons'/('C1_repaired_H3_S6_test50' if args.include_s6 else 'H3_vs_C1_repaired_test50_50k')
    SIZE=(4800 if args.include_s6 else 3200,960)
    movie_name='C1_repaired_H3_S6_test50.mp4' if args.include_s6 else 'C1_repaired_left_H3_right_test50.mp4'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'frames').mkdir(exist_ok=True)
    manifests=[SOURCE/name/'manifest.json' for name in ('C1_reused','H3_combined')]
    inputs=[read(p) for p in manifests]
    assert inputs[0]['source']==str(ROOT/'runs/pidg_color_repair/C1_repaired/rubber_duck')
    assert inputs[1]['source']==str(ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck')
    assert all(len(m['frames'])==50 for m in inputs)
    s6=None
    if args.include_s6:
        s6_path=ROOT/'tmp/pdfs/weekly_20261103/test_compare/evaluation/S6.json'
        s6=read(s6_path)
        assert s6['model']=='S6' and len(s6['frames'])==50
        assert s6['checkpoint']=='runs/pidg_staged_reconstruction/S6/formal/checkpoint.pt'
        assert s6['checkpoint_unchanged_after'] and s6['training_or_optimizer_updates']==0
        manifests.append(s6_path)

    def pair(index):
        records=[m['frames'][index] for m in inputs]
        for key in ('export_index','original_frame_id','normalized_time','filename','gt_sha256'):
            assert records[0][key]==records[1][key],(index,key)
        assert records[0]['export_index']==index
        frame=records[0]['original_frame_id']
        target=OUT/'frames'/f'test_{frame:04d}.png'
        panel=Image.new('RGB',SIZE,'#132638')
        draw=ImageDraw.Draw(panel)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
        source_receipts=[]
        panels=[(name,SOURCE/folder/'renders'/records[j]['filename'],records[j]['render_sha256'],'final 50k')
                for j,(folder,name) in enumerate((('C1_reused','C1_repaired'),('H3_combined','H3_combined')))]
        if s6 is not None:
            row=s6['frames'][index]
            assert row['frame']==frame and abs(row['time']-records[0]['normalized_time'])<1e-10
            path=ROOT/row['render']
            panels.append(('S6',path,s6['source_sha256'][str(path.relative_to(ROOT))],'final stage +4k | chain 46k'))
        for j,(name,path,checksum,budget) in enumerate(panels):
            assert sha(path)==checksum,path
            with Image.open(path) as image:
                image=image.convert('RGB')
                assert image.size==(1600,900)
                panel.paste(image,(j*1600,HEADER))
                pixels=image.tobytes()
            draw.text((j*1600+18,13),f'{name} | {budget} | test frame {frame:04d} | t={records[0]["normalized_time"]:.4f}',
                      font=font,fill='#f2f7fb')
            source_receipts.append({'model':name,'path':str(path.relative_to(ROOT)),
                                    'sha256':checksum,'pixel_sha256':hashlib.sha256(pixels).hexdigest()})
        panel.save(target,compress_level=3)
        with Image.open(target) as saved:
            for j,receipt in enumerate(source_receipts):
                crop=saved.crop((1600*j,HEADER,1600*(j+1),960)).tobytes()
                assert hashlib.sha256(crop).hexdigest()==receipt['pixel_sha256']
        return {'test_index':index,'original_frame_id':frame,'normalized_time':records[0]['normalized_time'],
                'file':str(target.relative_to(OUT)),'sha256':sha(target),'sources':source_receipts}

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(pair,range(50)))
    assert [r['original_frame_id'] for r in rows]==sorted(r['original_frame_id'] for r in rows)
    print('50 concatenated PNGs saved; all image regions verified pixel-exact.',flush=True)

    movie=OUT/movie_name
    writer=imageio_ffmpeg.write_frames(str(movie),SIZE,fps=FPS,codec='libx264',
        quality=9,macro_block_size=1,ffmpeg_timeout=60,
        output_params=['-preset','fast','-movflags','+faststart'])
    writer.send(None)
    try:
        for row in rows:
            with Image.open(OUT/row['file']) as image:writer.send(np.asarray(image.convert('RGB')))
    finally:writer.close()
    count,seconds=imageio_ffmpeg.count_frames_and_secs(movie)
    assert count==50 and abs(seconds-50/FPS)<.1
    # Decode the full movie to check that every frame is readable.
    decoded=imageio_ffmpeg.read_frames(str(movie),pix_fmt='rgb24')
    metadata=next(decoded)
    assert metadata['size']==SIZE
    decoded_count=sum(1 for _ in decoded)
    assert decoded_count==50

    preview_width=SIZE[0]//2
    preview=Image.new('RGB',(preview_width,1920),'white')
    for j,index in enumerate((0,16,33,49)):
        with Image.open(OUT/rows[index]['file']) as image:
            preview.paste(image.resize((preview_width,480),Image.Resampling.LANCZOS),(0,j*480))
    preview.save(OUT/'preview.png')

    files=[row['file'].replace('\\','/') for row in rows]
    ids=[row['original_frame_id'] for row in rows]
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C1_repaired | H3_combined - final 50k test renders</title>
<style>body{margin:20px;background:#132638;color:#f2f7fb;font:16px system-ui}h1{font-size:22px}img{display:block;width:100%;height:auto}input{width:min(65vw,900px)}button,a{margin:8px;color:#132638;background:#dbf1f0;padding:8px;border:0;border-radius:4px}a{display:inline-block;text-decoration:none}small{display:block;margin-bottom:12px;color:#c3d2df}</style>
<h1>Left: C1_repaired | Right: H3_combined</h1>
<small>Both final 50k checkpoints. All 50 original test frames, matched by frame ID and time. PNG image regions preserve source pixels; only labels are added. Playback is 5 test samples/s, not acquisition speed.</small>
<button id="play">Play</button><input id="slider" type="range" min="0" max="49" value="0"><span id="status"></span>
<a id="full" target="_blank">Open original-size PNG</a><a href="C1_repaired_left_H3_right_test50.mp4">Video</a>
<img id="comparison" alt="C1 repaired left and H3 right at the same test frame">
<script>const files=__FILES__,ids=__IDS__;const slider=document.getElementById('slider'),pic=document.getElementById('comparison'),status=document.getElementById('status'),full=document.getElementById('full'),play=document.getElementById('play');let timer=null;function show(){const i=+slider.value;pic.src=files[i];full.href=files[i];status.textContent=` ${i+1}/50 | frame ${String(ids[i]).padStart(4,'0')}`;}slider.oninput=show;play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='Play';}else{timer=setInterval(()=>{slider.value=(+slider.value+1)%50;show()},200);play.textContent='Pause';}};show();</script></html>'''
    html=html.replace('__FILES__',json.dumps(files)).replace('__IDS__',json.dumps(ids))
    if args.include_s6:
        html=html.replace('C1_repaired | H3_combined - final 50k test renders','C1_repaired | H3_combined | S6 - matched test renders')
        html=html.replace('Left: C1_repaired | Right: H3_combined','Left: C1_repaired | Middle: H3_combined | Right: S6')
        html=html.replace('Both final 50k checkpoints.','C1/H3: final 50k. S6: final +4k stage after 46k accepted-chain updates.')
        html=html.replace('C1 repaired left and H3 right at the same test frame','C1 repaired left, H3 middle and S6 right at the same test frame')
        html=html.replace('C1_repaired_left_H3_right_test50.mp4',movie_name)
    (OUT/'index.html').write_text(html,encoding='utf-8')
    description=('C1_repaired (left), H3_combined (middle), S6 (right).\nC1/H3: final 50k. S6: final +4k stage after 46k accepted-chain updates.\n'
                 if args.include_s6 else 'C1_repaired (left) vs H3_combined (right).\nBoth final 50k checkpoints.\n')
    (OUT/'README.txt').write_text(description+'Same 50 test frames, matched by original frame ID and normalized time.\nOpen index.html for the frame slider, frames/ for original-size concatenated PNGs, or the MP4 for playback.\nEach source render remains 1600x900; combined PNGs are '+str(SIZE[0])+'x960 including a 60-pixel label bar.\nPNG render regions are pixel-exact. The MP4 is a compressed viewing copy, shown at 5 test samples/s rather than acquisition speed.\nNo new training or inference was performed.\n',encoding='utf-8')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'split':'test','checkpoints_iteration':50000,
        'panel_order':['C1_repaired','H3_combined']+(['S6'] if args.include_s6 else []),'image_count':50,'source_size':[1600,900],
        'paired_size':list(SIZE),'header_height':HEADER,'source_pixels_preserved':True,
        'training_or_new_inference':False,'source_manifests':{str(p.relative_to(ROOT)):sha(p) for p in manifests},
        'video':{'file':movie.name,'sha256':sha(movie),'fps':FPS,'decoded_frames':decoded_count,
                 'duration_seconds':seconds,'viewing_copy':True,'acquisition_speed':False},'frames':rows}
    if s6 is not None:
        manifest['checkpoints_iteration']={'C1_repaired':50000,'H3_combined':50000,'S6_stage':4000,'S6_accepted_chain':46000}
        manifest['S6_checkpoint_receipt']={'checkpoint':s6['checkpoint'],'sha256':s6['checkpoint_sha256']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=OUT.parent/(OUT.name+'.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len([name for name in z.namelist() if name.startswith('frames/') and name.endswith('.png')])==50
    print(json.dumps({'output':str(OUT),'archive':str(archive),'archive_MB':round(archive.stat().st_size/1e6,1),
                      'video_frames':decoded_count,'video_seconds':seconds,'all_source_pixels_preserved':True}),flush=True)


if __name__=='__main__':main()
