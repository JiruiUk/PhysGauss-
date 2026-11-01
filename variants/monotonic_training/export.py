"""Pixel-identity-checked GT|Render PNGs, videos, gallery and final metrics."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import subprocess
import zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'experiments/monotonic_training'
REFERENCE=ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def video(pairs,path):
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    command=[ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','3200x950','-r','10','-i','-',
             '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    with path.with_suffix('.encode.log').open('wb') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for p in pairs:
                with Image.open(p) as im:proc.stdin.write(im.convert('RGB').tobytes())
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('Video encoding failed')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait()
    count,seconds=imageio_ffmpeg.count_frames_and_secs(str(path))
    if count!=len(pairs) or abs(seconds-len(pairs)/10)>.11:raise ValueError('Video frame count/duration mismatch')
    return {'file':str(path),'frames':count,'display_fps':10,'seconds':seconds,'note':'display rate only; acquisition FPS unknown'}


def export(directory):
    directory=Path(directory);out=ROOT/'output/monotonic_training'/directory.parent.name
    out.mkdir(parents=True,exist_ok=True)
    if not (directory.parents[1]/'selection_lock.json').exists():raise ValueError('Final test selection must be locked')
    lock=read(directory.parents[1]/'selection_lock.json')
    frames=read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')['frames']
    # Scene's shuffle=False render loader uses the original metadata order.
    names=[int(Path(f['file_path']).stem) for f in frames]
    if names!=sorted(names) or len(names)!=50:raise ValueError('Unexpected final-test frame order')
    metrics=read(directory/'results.json');per=read(directory/'per_view.json')
    reference=read(REFERENCE/'results.json')['ours_50000'];reference_per=read(REFERENCE/'per_view.json')['ours_50000']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
    def pair(task):
        tag,i=task;name=f'{i:05d}.png';raw=directory/'test'/('ours_'+tag)
        with Image.open(raw/'gt'/name) as ag,Image.open(raw/'renders'/name) as ar:
            gt,render=ag.convert('RGB'),ar.convert('RGB')
            if gt.size!=render.size or gt.size!=(1600,900):raise ValueError('Unexpected output dimensions')
            with Image.open(REFERENCE/'test/ours_50000/gt'/name) as ref:
                if gt.tobytes()!=ref.convert('RGB').tobytes():raise ValueError('Reference GT mismatch')
            canvas=Image.new('RGB',(3200,950),'#edf2f7');draw=ImageDraw.Draw(canvas)
            draw.text((16,10),f'Ground Truth | frame {names[i]:04d} | t={(names[i]-1)/199:.6f}',font=font,fill='#17304a')
            draw.text((1616,10),f'Render | {directory.parent.name} | {tag}',font=font,fill='#17304a')
            canvas.paste(gt,(0,50));canvas.paste(render,(1600,50))
            dest=out/'pairs'/tag/name;dest.parent.mkdir(parents=True,exist_ok=True);canvas.save(dest,compress_level=3)
            with Image.open(dest) as check:
                if check.crop((0,50,1600,950)).tobytes()!=gt.tobytes() or check.crop((1600,50,3200,950)).tobytes()!=render.tobytes():raise ValueError('Concatenation changed image pixels')
            mse=np.mean(((np.asarray(gt,dtype=np.float32)-np.asarray(render,dtype=np.float32))/255)**2,dtype=np.float64)
            score=-10*math.log10(max(mse,1e-12));saved=per['ours_'+tag]['PSNR'][name]
            if abs(score-saved)>2e-4:raise ValueError('Saved PNG PSNR mismatch')
            return {'snapshot':tag,'index':i,'frame_id':names[i],'time':(names[i]-1)/199,'pair':str(dest),'PSNR':score,'saved_score_difference':score-saved}
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(pair,[(tag,i) for tag in ['best','50000'] for i in range(50)]))
    videos={tag:video([out/'pairs'/tag/f'{i:05d}.png' for i in range(50)],out/f'GT_Render_{tag}.mp4') for tag in ['best','50000']}
    deltas={}
    for tag in ['best','50000']:
        values=[per['ours_'+tag]['PSNR'][f'{i:05d}.png']-reference_per['PSNR'][f'{i:05d}.png'] for i in range(50)]
        deltas[tag]={'mean_PSNR_delta_vs_H3_final':statistics.mean(values),'median_PSNR_delta':statistics.median(values),
          'improved_frames':sum(x>0 for x in values),'worsened_frames':sum(x<0 for x in values),
          'worst5_PSNR':statistics.mean(sorted(per['ours_'+tag]['PSNR'].values())[:5])}
    diagnostic=[json.loads(s) for s in (directory/'kinematic_diagnostics.jsonl').read_text().splitlines()][-1]
    diagnostic_mean=statistics.mean(p['relative_rmse'] for p in diagnostic['times'])
    summary={'model':directory.parent.name,'source':str(directory),'completed_iteration':50000,'selection':lock,
      'clock_mode':'fixed_monotonic','kinematic_weight':.001,'test_metrics':metrics,'H3_final_reference':reference,
      'paired_test_comparison':deltas,'final_internal_velocity_relative_rmse':diagnostic_mean,
      'videos':videos,'pair_count':len(rows),'pixel_identity_verified':True,'maximum_independent_PSNR_difference':max(abs(r['saved_score_difference']) for r in rows),
      'frame_mapping':rows,'image_shape':[3200,950],'split':[120,30,50],
      'limitations':['single scene/seed','clock and kinematic loss jointly changed','historical test exposure','internal velocity consistency is not physical GT']}
    write(out/'manifest.json',summary);write(REPORTS/'results.json',summary)
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Ground Truth | Render</title><style>body{font:17px system-ui;background:#112035;color:#eef4fc;margin:24px}img,video{max-width:100%}a{color:#9fd7ff}select,input{margin:12px}</style><h1>Ground Truth | Render</h1><p>左：Ground Truth；右：Render。50个测试帧，PNG图像区域逐像素核验。视频10fps仅作展示。</p><select id="tag"><option value="best">验证集PSNR选择</option><option value="50000">最终50,000步</option></select><input id="frame" type="range" min="0" max="49" value="21"><span id="number"></span><p><a id="full">完整PNG</a> | <a id="movie">下载MP4</a></p><img id="image"><video id="video" controls loop></video><script>const t=document.getElementById("tag"),f=document.getElementById("frame"),ids='+json.dumps(names)+';function update(){const n=String(f.value).padStart(5,"0"),p="pairs/"+t.value+"/"+n+".png";document.getElementById("image").src=p;document.getElementById("full").href=p;document.getElementById("number").textContent="原帧 "+ids[Number(f.value)]}function change(){update();const p="GT_Render_"+t.value+".mp4";document.getElementById("video").src=p;document.getElementById("movie").href=p}t.onchange=change;f.oninput=update;change()</script>'
    (out/'index.html').write_text(page,encoding='utf8')
    lines=['# 单调物理时钟 + 速度—轨迹一致性：完整训练结果','',
      '模型：J1_monotonic_kinematic。Rubber Duck，seed0，120训练/30验证/50历史已暴露测试。',
      '复用与H3相同的完整3k起点，继续训练至总50k，不是从收敛H3再追加50k。',
      '原PIDG代码与H3权重未修改；新分支保留原图像/覆盖/邻边应变/CMR，加入固定单调速度时钟与0.001联合运动学一致性。',
      '数据帧编号/共同时间轴/原相机采样器未改；没有外力/本构/自由材料参数。','',
      f"验证PSNR选中第{lock['best']['iteration']}步；选择在打开最终测试渲染前锁定。最终50k也完整导出。",'',
      '| 快照 | PSNR↑ | SSIM↑ | LPIPS↓ | MS-SSIM↑ | 相对H3最终PSNR | 改善帧/50 |',
      '|---|---:|---:|---:|---:|---:|---:|',
      f"| H3已有最终50k | {reference['PSNR']:.5f} | {reference['SSIM']:.7f} | {reference['LPIPS']:.7f} | {reference['MS-SSIM']:.7f} | - | - |"]
    for tag in ['best','50000']:
        m=metrics['ours_'+tag];q=deltas[tag]
        lines.append(f"| {tag} | {m['PSNR']:.5f} | {m['SSIM']:.7f} | {m['LPIPS']:.7f} | {m['MS-SSIM']:.7f} | {q['mean_PSNR_delta_vs_H3_final']:+.5f} | {q['improved_frames']} |")
    lines+=['',f'最终内部速度—轨迹相对RMSE：{diagnostic_mean*100:.2f}%（5个训练探针平均；非物理速度真值）。',
      '', '时钟与一致性同时改变，因此图像差异不能全部归因为时钟修正；也不能由自一致性改善推断材料或外力识别成功。',
      '', '## Ground Truth / Render 导出','',
      f'- [交互浏览全部帧]({(out/"index.html").as_posix()})',
      f'- [验证选择模型拼接视频]({(out/"GT_Render_best.mp4").as_posix()})',
      f'- [最终50k拼接视频]({(out/"GT_Render_50000.mp4").as_posix()})',
      f'- [原帧85的完整拼接PNG]({(out/"pairs/best/00021.png").as_posix()})',
      '',f'共{len(rows)}张3200×950拼接PNG，两段50帧视频；各PNG左右图像区与原GT/render逐像素一致，独立重算PSNR最大差{summary["maximum_independent_PSNR_difference"]:.9f}dB。',
      '', '## 原始证据','', '[训练与导出协议](plan.json) | [预检](preflight.json) | [源码冻结](source_freeze.json) | [结果JSON](results.json)',
      '', '最终测试有历史调参暴露；本轮仅用验证选择，但不把它描述为全新未暴露测试集。']
    (REPORTS/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    (out/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    archive=out.parent/(out.name+'_GT_Render.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.endswith('.encode.log'):z.write(p,p.relative_to(out))
    write(REPORTS/'exports.json',{'gallery':str(out/'index.html'),'best_video':str(out/'GT_Render_best.mp4'),
      'final_video':str(out/'GT_Render_50000.mp4'),'archive':str(archive),'pairs':str(out/'pairs'),'pixel_identity_verified':True})
    print('EXPORTED '+str(out),flush=True)


if __name__=='__main__':
    import sys
    export(Path(sys.argv[1]))
