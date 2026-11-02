"""Verified GT|Render artifacts and matched-arm comparison."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from variants.monotonic_training.export import video
from variants.metric_tuning.run import read_json as read, write_json as write
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'experiments/flow_comparison'
RUNS = ROOT/'runs/pidg_flow_comparison'
OUT = ROOT/'output/flow_comparison'


def export(directory):
    name = directory.parent.name; out = OUT/name
    lock = read(RUNS/'selection_lock.json')
    if lock['selection_split'] != 'validation' or name not in lock['models']:
        raise ValueError('Missing locked validation selection')
    frame_ids = [int(Path(f['file_path']).stem) for f in read(ROOT/'data/derived/kinematic_resplit/final_test/transforms_test.json')['frames']]
    per = read(directory/'per_view.json'); font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
    def pair(task):
        tag, i = task; filename = f'{i:05d}.png'; source = directory/'test'/('ours_'+tag)
        with Image.open(source/'gt'/filename) as a, Image.open(source/'renders'/filename) as b:
            gt, render = a.convert('RGB'), b.convert('RGB')
            if gt.size != (1600, 900) or render.size != gt.size:
                raise ValueError('Wrong output dimensions')
            reference = ROOT/'runs/pidg_integrity_tuning/H3_combined/rubber_duck/test/ours_50000/gt'/filename
            with Image.open(reference) as previous:
                if previous.convert('RGB').tobytes() != gt.tobytes():
                    raise ValueError('Ground truth mismatch')
            canvas = Image.new('RGB', (3200, 950), '#edf2f7'); draw = ImageDraw.Draw(canvas)
            draw.text((16, 10), f'Ground Truth | frame {frame_ids[i]:04d} | t={(frame_ids[i]-1)/199:.6f}', font=font, fill='#17304a')
            draw.text((1616, 10), f'Render | {name} | {tag}', font=font, fill='#17304a')
            canvas.paste(gt, (0, 50)); canvas.paste(render, (1600, 50))
            destination = out/'pairs'/tag/filename; destination.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(destination, compress_level=3)
            with Image.open(destination) as check:
                if check.crop((0, 50, 1600, 950)).tobytes() != gt.tobytes() or check.crop((1600, 50, 3200, 950)).tobytes() != render.tobytes():
                    raise ValueError('Concatenation pixel mismatch')
            mse = np.mean((np.asarray(gt, dtype=np.float64)/255-np.asarray(render, dtype=np.float64)/255)**2)
            psnr = -10*math.log10(max(mse, 1e-12))
            difference = abs(psnr-per['ours_'+tag]['PSNR'][filename])
            if difference > 1e-4:
                raise ValueError('Independent PSNR mismatch')
            return difference
    with ThreadPoolExecutor(max_workers=6) as pool:
        differences = list(pool.map(pair, [(tag, i) for tag in ['best', '50000'] for i in range(50)]))
    movies = [video(sorted((out/'pairs'/tag).glob('*.png')), out/('GT_Render_'+tag+'.mp4')) for tag in ['best', '50000']]
    page = '<!doctype html><meta charset="utf-8"><title>'+name+'</title><style>body{font:18px system-ui;background:#112035;color:white;margin:24px}img,video{max-width:100%}</style><h1>'+name+'</h1><p>左：Ground Truth；右：Render。50个测试帧；视频10fps仅为播放速度。</p><video controls loop src="GT_Render_best.mp4"></video><p><a href="GT_Render_50000.mp4">最终50k视频</a></p><input id="f" type="range" min="0" max="49" value="21"><img id="im"><script>const f=document.getElementById("f");function u(){document.getElementById("im").src="pairs/best/"+String(f.value).padStart(5,"0")+".png"}f.oninput=u;u()</script>'
    (out/'index.html').write_text(page, encoding='utf8')
    manifest = {'model': name, 'selection': lock['models'][name], 'metrics': read(directory/'results.json'),
        'pairs': 100, 'pixel_identity_verified': True, 'independent_PSNR_max_difference': max(differences),
        'videos': movies, 'velocity_source': read(directory/'metric_config.json')['velocity_source']}
    write(out/'manifest.json', manifest)
    archive = OUT/(name+'_GT_Render.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.endswith('.encode.log'):
                z.write(p, p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
    write(REPORTS/(name+'_exports.json'), {**manifest, 'gallery': str(out/'index.html'), 'archive': str(archive)})


def report():
    names = ['K1_flow_independent', 'K2_flow_trajectory']
    rows = []
    for name in names:
        d = RUNS/name/'rubber_duck'; metrics = read(d/'results.json')
        for tag in ('best', '50000'):
            rows.append({'model': name, 'snapshot': tag, **metrics['ours_'+tag]})
    references = []
    for parent, name in [('pidg_integrity_tuning', 'H3_combined'), ('pidg_monotonic_training', 'J1_monotonic_kinematic')]:
        references.append({'model': name, 'snapshot': '50000', **read(ROOT/'runs'/parent/name/'rubber_duck/results.json')['ours_50000']})
    write(REPORTS/'results.json', {'rows': rows, 'historical_references': references,
        'protocol': read(REPORTS/'plan.json'), 'selection_lock': read(RUNS/'selection_lock.json')})
    lines = ['# 光流与轨迹导出速度：完整训练对照', '',
        'Rubber Duck，seed0；120训练/30验证/50测试。同一完整3k起点训练至总50k。',
        'K1：独立速度头 + 运动光流。K2：轨迹有限差分速度 + 相同运动光流。两组共用单调时钟、局部材料坐标动量残差和图像损失。',
        '光流与相机补偿深度为估计量；仅120训练图像生成监督。光流400×225，RGB1600×900。', '',
        '| 模型 | 快照 | PSNR↑ | SSIM↑ | LPIPS↓ | MS-SSIM↑ |', '|---|---|---:|---:|---:|---:|']
    for row in references+rows:
        lines.append('| '+row['model']+' | '+row['snapshot']+' | '+' | '.join(f'{row[k]:.7f}' for k in ['PSNR', 'SSIM', 'LPIPS', 'MS-SSIM'])+' |')
    lines += ['', '两组新模型之间只改变速度来源。相对历史H3/J1还共同改变了光流、时钟或动量离散化，不能把差异全归于光流。',
        'K2内部速度误差为零是结构定义，不能作为物理预测正确的证据。没有增加外力、本构或自由材料参数。',
        '本轮只用验证选择；最终测试此前有历史调参暴露，不能称全新未暴露测试。单场景、单随机种子的结论有限。', '', '## GT / Render', '']
    for name in names:
        out = OUT/name
        lines += [f'- [{name} 浏览全部帧]({(out/"index.html").as_posix()})',
            f'- [{name} 验证选择拼接视频]({(out/"GT_Render_best.mp4").as_posix()})',
            f'- [{name} 完整压缩包]({(OUT/(name+"_GT_Render.zip")).as_posix()})']
    (REPORTS/'assessment.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
