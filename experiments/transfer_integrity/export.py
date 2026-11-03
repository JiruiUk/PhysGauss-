"""Verify all temporal videos and make an honest comparison gallery."""
from pathlib import Path
import sys
import html
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning.runtime import read_json,write_json,sha256
from variants.view_completeness.metrics import evaluate,summarize,shape_proxies
OUT=ROOT/'output/transfer_integrity';REPORT=ROOT/'experiments/transfer_integrity'
NAMES=['C1_original_50k','U0_C1_control','U1_transfer_geometry']
VIEWS=['fixed_0002','fixed_0086','fixed_0188','moving_control']
SELECTED=[2,27,43,74,86,108,172,200]


def main():
    proof=read_json(REPORT/'render_verification.json')
    for p,h in proof['checkpoint_sha256_unchanged'].items():assert sha256(Path(p))==h
    for p,h in proof['source_sha256_unchanged'].items():assert sha256(ROOT/p)==h
    summary={};per_frame={}
    for name in NAMES:
        result=read_json(REPORT/f'{name}_results.json')
        assert result['fixed_camera_fingerprints_before']==result['fixed_camera_fingerprints_after']
        for view in VIEWS:
            base=OUT/name/view;rows=[];yellow=[];blue=[]
            assert len(list((base/'small').glob('rgb*.png')))==200
            assert len(list((base/'small').glob('alpha*.png')))==200
            for n in range(1,201):
                a=np.asarray(Image.open(base/'small'/f'alpha{n:04d}.png'),dtype=float)/255
                rgb=np.asarray(Image.open(base/'small'/f'rgb{n:04d}.png'),dtype=float)/255
                row={'frame':n,**evaluate(a,enabled=True)};rows.append(row)
                # These rendered color subsets are appearance proxies only.
                y=(np.minimum(rgb[...,0],rgb[...,1])>rgb[...,2]*1.35+.03)&(rgb[...,0]>.15)&(rgb[...,1]>.13)&(a>.5)
                b=(rgb[...,2]>rgb[...,0]*1.3+.02)&(rgb[...,2]>rgb[...,1]*1.05)&(a>.5)
                yellow.append({'frame':n,'shape_proxies':shape_proxies(y.astype(float))})
                blue.append({'frame':n,'shape_proxies':shape_proxies(b.astype(float))})
            for n in SELECTED:
                with Image.open(base/'native'/f'rgb{n:04d}.png') as im:
                    assert im.size==(1600,900)
                    assert np.array_equal(np.asarray(im.resize((400,225),Image.Resampling.LANCZOS)),np.asarray(Image.open(base/'small'/f'rgb{n:04d}.png')))
            key=f'{name}/{view}'
            def describe(items):
                s=summarize(items);areas=[x['shape_proxies']['foreground_pixels'] for x in items]
                s['area_mean']=float(np.mean(areas));s['area_min']=int(min(areas));s['area_max']=int(max(areas))
                return s
            summary[key]={'all_alpha':describe(rows),'yellow_appearance':describe(yellow),'blue_appearance':describe(blue)}
            per_frame[key]={'alpha':rows,'yellow':yellow,'blue':blue}
    write_json(REPORT/'shape_summary.json',summary);write_json(REPORT/'shape_frames.json',per_frame)
    decoded={}
    for p in sorted(OUT.rglob('*.mp4')):
        gen=imageio_ffmpeg.read_frames(str(p),pix_fmt='rgb24');meta=next(gen);count=0;chosen=[]
        size=(1200,510) if 'comparison' in p.name else (800,480)
        assert tuple(meta['size'])==size and abs(meta['fps']-25)<1e-8
        for frame in gen:
            count+=1
            if 'comparison' in p.name and count in SELECTED:
                arr=np.frombuffer(frame,np.uint8).reshape(size[1],size[0],3)
                chosen.append((count,Image.fromarray(arr.copy())))
        assert count==200,(p,count)
        if chosen:
            contact=Image.new('RGB',(1200,8*252),'#171a21');draw=ImageDraw.Draw(contact)
            for i,(n,im) in enumerate(chosen):
                draw.text((8,i*252+5),f'DECODED | {p.stem} | frame {n}',fill='white')
                contact.paste(im.crop((0,30,1200,255)),(0,i*252+27))
            contact.save(p.with_suffix('.decoded_contact.png'))
        decoded[str(p.relative_to(OUT))]={'frames':count,'dimensions':size,'fps':meta['fps'],'sha256':sha256(p)}
        print('DECODED',p.name,count,flush=True)
    assert len(decoded)==16
    write_json(REPORT/'video_verification.json',decoded)
    # Three models at the same fixed camera; no ground-truth label here.
    for view in VIEWS:
        canvas=Image.new('RGB',(1200,3*252),'#171a21');draw=ImageDraw.Draw(canvas)
        for y,n in enumerate([2,86,200]):
            for x,name in enumerate(NAMES):
                draw.text((400*x+6,252*y+5),f'{name} | {view} | time {n}',fill='white')
                canvas.paste(Image.open(OUT/name/view/'small'/f'rgb{n:04d}.png'),(400*x,252*y+27))
        canvas.save(OUT/f'{view}_key_comparison.png')
    parts=['<!doctype html><meta charset="utf-8"><title>Transfer integrity 第一版测试</title>',
           '<style>body{background:#11151c;color:#eee;font:17px system-ui;margin:26px;max-width:1600px}img,video{max-width:100%}a{color:#8bc9ff}section{border-top:1px solid #555;margin-top:30px}</style>',
           '<h1>第一版：真实观测、短时对应与局部几何先验</h1>',
           '<p>左：原始 C1 50k；中：继续训练的 U0 52k；右：新约束 U1 52k。固定相机与原相机轨迹各200时刻。视频上排RGB、下排透明度。</p>',
           '<p>由最终完整检查点在1600×900渲染后缩图。只有原轨迹存在图像真值；固定相机跨时刻结果不能计算真实PSNR或证明物理身份。黄色/蓝色只用于外观诊断。25fps为展示速度。</p>',
           '<p><a href="../../experiments/transfer_integrity/assessment.md">中文测试结论</a></p>']
    for view in VIEWS:
        parts.extend([f'<section><h2>{view}</h2>',f'<video controls loop preload="metadata" src="{view}_comparison_RGB_alpha.mp4"></video>',
                      f'<p><a href="{view}_comparison_RGB_alpha.decoded_contact.png">逐帧解码接触表</a></p>',
                      f'<img loading="lazy" src="{view}_key_comparison.png">'])
        for name in NAMES:
            parts.extend([f'<details><summary>{name} 原图、视频与透明度</summary>',
                          f'<video controls loop preload="none" src="{name}/{view}_RGB.mp4"></video>',
                          f'<img loading="lazy" src="{name}/{view}_RGB_alpha_contact.png">','<p>1600×900原图：'])
            for n in SELECTED:
                parts.extend([f'<a href="{name}/{view}/native/rgb{n:04d}.png">{n} RGB</a> / ',f'<a href="{name}/{view}/native/alpha{n:04d}.png">alpha</a>　'])
            parts.append('</p></details>')
        parts.append('</section>')
    (OUT/'index.html').write_text('\n'.join(parts)+'\n',encoding='utf-8')
    write_json(REPORT/'artifact_manifest.json',{str(p.relative_to(ROOT)):sha256(p) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    write_json(REPORT/'export_verification.json',{'videos_fully_decoded':16,'frames_per_video':200,'native_to_small_RGB_exact':True,
               'fixed_camera_fingerprints_unchanged':True,'checkpoints_and_render_sources_unchanged':True,
               'novel_camera_time_GT_available':False,'exporter_sha256':sha256(Path(__file__))})


if __name__=='__main__':main()
