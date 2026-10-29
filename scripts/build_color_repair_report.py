"""Evidence-based Chinese report for repaired C1/C2/C3, fixed50 final50k."""
from datetime import datetime, timezone
from html import unescape
import hashlib
import json
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import minimum_filter, label, find_objects
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import pidg_ablation_report_layout as L

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
PDF = OUT / 'PIDG_Color_Repair_Ablation_Report_CN.pdf'
ASSETS = OUT / 'color_repair_assets'
REPORTS = ROOT / 'experiments/color_repair'
RUNS = ROOT / 'runs/pidg_color_repair'
OLD = ROOT / 'runs/pidg_coverage_tuning'
EXPORTS = RUNS / 'render_exports/final50000'
METRICS = ('PSNR', 'SSIM', 'LPIPS', 'MS-SSIM')
NAMES = ('C1_repaired', 'C2_repaired', 'C3_repaired')
OLD_NAMES = ('C0_V3_reference', 'C1_mask', 'C2_mask_affine', 'C3_mask_affine_density')
COLORS = ['#087f8c', '#d48a36', '#7564a8']
SOURCES, RECORD = {}, []
PAGES = 12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = digest(path)
    return path


def read(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(s) for s in track(path).read_text().splitlines() if s.strip()]


def rgb(path):
    with Image.open(track(path)) as im:
        return np.array(im.convert('RGB'))


def point_count(directory):
    path = track(directory / 'point_cloud/iteration_50000/point_cloud.ply')
    with path.open('rb') as stream:
        for line in stream:
            if line.startswith(b'element vertex'):
                return int(line.split()[-1])
    raise ValueError('Missing Gaussian count')


def load():
    assert read(RUNS / 'controller.json')['status'] == 'completed'
    plan = read(REPORTS / 'plan.json')
    init = read(REPORTS / 'initialization.json')
    assert init['iteration'] == 0 and not init['pretrained_model_used']
    metadata = read(ROOT / 'data/derived/kinematic_resplit/final_test/transforms_test.json')
    assert [int(Path(f['file_path']).name) for f in metadata['frames']] == list(range(1, 201, 4))
    assessment = read(ROOT / 'experiments/coverage_isolated/assessment.json')
    previous = {m['name']: m for m in assessment['models']}
    models = []
    for name in (*OLD_NAMES, *NAMES):
        is_new = name in NAMES
        directory = (RUNS if is_new else OLD) / name / 'rubber_duck'
        scores = read(directory / 'results.json')['ours_50000']
        per_view = read(directory / 'per_view.json')['ours_50000']
        coverage = read(directory / 'coverage_metrics_test.json')['50000']
        config = read(directory / 'metric_config.json')
        best = read(directory / 'metric_best.json')
        assert all(v['selection_split'] == 'validation' for v in best.values())
        for key in METRICS:
            assert set(per_view[key]) == {f'{i:05d}.png' for i in range(50)}
            assert all(math.isfinite(v) for v in per_view[key].values())
            assert abs(math.fsum(per_view[key].values()) / 50 - scores[key]) < 1e-5
        events = lines(directory / 'coverage_density.jsonl') if (directory / 'coverage_density.jsonl').exists() else []
        models.append({'id': ('N' if is_new else 'O') + str(NAMES.index(name) + 1 if is_new else OLD_NAMES.index(name)),
                       'label': ('新C' + str(NAMES.index(name) + 1)) if is_new else ('C0/V3' if name == OLD_NAMES[0] else '旧C' + str(OLD_NAMES.index(name))),
                       'name': name, 'directory': str(directory), 'scores': scores, 'per_view': per_view,
                       'coverage': coverage, 'config': config, 'best': best, 'points': point_count(directory),
                       'events': events, 'history': lines(directory / 'metric_history.jsonl'),
                       'alpha': None if is_new else previous[name]['alpha_diagnostics']})
    branches = [read(RUNS / n / 'rubber_duck/branch.json') for n in NAMES]
    middle = read(RUNS / 'affine_prefix/rubber_duck/branch.json')
    assert branches[0]['iteration'] == middle['iteration'] == 3000
    assert branches[0]['identical_full_checkpoint_sha256'] == middle['identical_full_checkpoint_sha256']
    assert branches[1]['iteration'] == branches[2]['iteration'] == 15000
    assert branches[1]['identical_full_checkpoint_sha256'] == branches[2]['identical_full_checkpoint_sha256']
    b, c = models[5]['config'], models[6]['config']
    assert {k for k in b if b[k] != c[k]} == {'coverage_density'}
    selected = read(RUNS / 'selection_lock.json')
    assert selected['validation_selections'] == {m['name']: m['best'] for m in models[4:]}
    for i in range(50):
        gt_hashes = [digest(track(Path(m['directory']) / f'test/ours_50000/gt/{i:05d}.png')) for m in models]
        assert len(set(gt_hashes)) == 1
    protocol = {'scene': 'Rubber Duck', 'seed': 0, 'split': [120, 30, 50], 'iterations_each': 50000,
                'render_size': [1600, 900], 'selection': 'validation only', 'snapshot': 'final50000',
                'historical_test_exposure': True, 'interleaved_time_interpolation_not_forecasting': True,
                'all_seven_GT_exports_identical': True, 'initialization': init,
                'shared3k_sha256': middle['identical_full_checkpoint_sha256'],
                'shared15k_C2_C3_sha256': branches[1]['identical_full_checkpoint_sha256'],
                'C2_C3_only_config_difference': 'coverage_density'}
    return metadata, models, protocol, plan


def alpha_audit(metadata, models):
    for m in models[4:]:
        m['alpha'] = {'per_frame': [], 'mean': {}}
    for i, frame in enumerate(metadata['frames']):
        source = ROOT / 'data/derived/kinematic_resplit/final_test' / (frame['file_path'] + '.png')
        with Image.open(track(source)) as im:
            target = np.array(im.getchannel('A').resize((1600, 900), Image.Resampling.BICUBIC)) / 255.
        truth = target > .5
        core = minimum_filter(target, size=7, mode='constant', cval=0) > .95
        for m in models[4:]:
            path = Path(m['directory']) / f'test/ours_50000/alpha/{i:05d}.png'
            with Image.open(track(path)) as im:
                a = np.array(im.convert('L')) / 255.
            pred = a > .5
            tp, fp, fn = [int(x.sum()) for x in (truth & pred, ~truth & pred, truth & ~pred)]
            low = int((core & (a < .5)).sum())
            stored = m['coverage']['per_frame'][i]
            assert int(stored['frame']) == int(Path(frame['file_path']).name)
            assert stored['interior_pixels'] == int(core.sum())
            assert abs(stored['interior_hole_rate'] - low / max(int(core.sum()), 1)) < 1e-12
            assert abs(stored['mask_iou'] - tp / max(tp + fp + fn, 1)) < 1e-8
            m['alpha']['per_frame'].append({'frame_id': int(stored['frame']), 'mask_iou': stored['mask_iou'],
                'background_false_positive_rate': fp / max(int((~truth).sum()), 1),
                'background_false_positive_pixels': fp, 'background_pixels': int((~truth).sum()),
                'interior_low_coverage_pixels': low, 'interior_pixels': int(core.sum())})
    for m in models[4:]:
        for key in ('mask_iou', 'background_false_positive_rate'):
            m['alpha']['mean'][key] = float(np.mean([row[key] for row in m['alpha']['per_frame']]))


def compare(a, b):
    result = {'metric_delta': {k: a['scores'][k] - b['scores'][k] for k in METRICS}, 'per_frame': {}}
    for k in METRICS:
        d = np.array([a['per_view'][k][f'{i:05d}.png'] - b['per_view'][k][f'{i:05d}.png'] for i in range(50)])
        result['per_frame'][k] = {'mean': float(d.mean()), 'median': float(np.median(d)),
                                  'improved_frames': int((d < 0 if k == 'LPIPS' else d > 0).sum()),
                                  'min': float(d.min()), 'max': float(d.max())}
    return result


def charts(models):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    paths = {}
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), constrained_layout=True)
    for ax, key in zip(axes.flat, METRICS):
        x = np.arange(3)
        old = [m['scores'][key] for m in models[1:4]]
        new = [m['scores'][key] for m in models[4:]]
        ax.bar(x - .18, old, .36, label='Historical', color='#aab5c0')
        ax.bar(x + .18, new, .36, label='Repaired', color=COLORS)
        ax.axhline(models[0]['scores'][key], color='#42576b', ls='--', lw=1.2, label='C0 / V3 reference')
        lo, hi = min(old + new + [models[0]['scores'][key]]), max(old + new + [models[0]['scores'][key]])
        gap = max((hi - lo) * .18, 1e-5)
        ax.set_ylim(lo - gap, hi + gap)
        ax.set_xticks(x, ['C1', 'C2', 'C3']); ax.set_title(key + (' (lower better)' if key == 'LPIPS' else ' (higher better)'))
        ax.grid(axis='y', alpha=.15)
    axes[0, 0].legend(fontsize=7, loc='lower right')
    paths['metric_bars'] = ASSETS / 'metric_bars.png'
    fig.savefig(paths['metric_bars'], dpi=190); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
    ids = np.arange(1, 201, 4)
    for m, color in zip(models[4:], COLORS):
        axes[0].plot(ids, [m['per_view']['PSNR'][f'{i:05d}.png'] for i in range(50)], label=m['id'], color=color)
    axes[0].set_ylabel('Test PSNR (dB)'); axes[0].legend(ncol=3); axes[0].grid(alpha=.2)
    delta = np.array([models[6]['per_view']['PSNR'][f'{i:05d}.png'] - models[5]['per_view']['PSNR'][f'{i:05d}.png'] for i in range(50)])
    axes[1].bar(ids, delta, width=2.8, color=np.where(delta >= 0, '#087f8c', '#d48a36'))
    axes[1].axhline(0, color='#42576b', lw=.8); axes[1].set_ylabel('N3 - N2 PSNR (dB)')
    axes[1].set_xlabel('Original frame ID (fixed test set)'); axes[1].grid(axis='y', alpha=.2)
    paths['per_frame'] = ASSETS / 'per_frame.png'
    fig.savefig(paths['per_frame'], dpi=190); plt.close(fig)
    events = models[6]['events']
    fig, ax = plt.subplots(figsize=(9, 2.5), constrained_layout=True)
    ax.bar([e['iteration'] / 1000 for e in events], [e['added'] for e in events], width=1.2, color=COLORS[2])
    ax.set_xlabel('Training iteration (k)'); ax.set_ylabel('New Gaussians'); ax.set_xticks(range(18, 37, 2)); ax.grid(axis='y', alpha=.2)
    paths['density'] = ASSETS / 'density_events.png'
    fig.savefig(paths['density'], dpi=190); plt.close(fig)
    return paths


def montage(path, arrays, labels, cell=440):
    h, w = arrays[0].shape[:2]
    target_h = round(cell * h / w)
    image = Image.new('RGB', (cell * len(arrays), target_h + 55), '#edf2f7')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    for i, (array, text) in enumerate(zip(arrays, labels)):
        draw.text((i * cell + 12, 12), text, font=font, fill='#17304a')
        image.paste(Image.fromarray(array).resize((cell, target_h), Image.Resampling.LANCZOS), (i * cell, 55))
    image.save(path)
    return path


def visual_cases(metadata, models):
    info, paths = {}, {}
    for index in (0, 38):
        raw = f'test/ours_50000/gt/{index:05d}.png'
        gt = rgb(Path(models[4]['directory']) / raw)
        mask = (gt.max(-1) > 30)
        y, x = np.where(mask)
        box = (max(0, int(x.min()) - 25), max(0, int(y.min()) - 25), min(1600, int(x.max()) + 26), min(900, int(y.max()) + 26))
        x0, y0, x1, y1 = box
        arrays = [gt[y0:y1, x0:x1]] + [rgb(Path(m['directory']) / f'test/ours_50000/renders/{index:05d}.png')[y0:y1, x0:x1] for m in models[4:]]
        path = ASSETS / f'frame_{index:05d}_subject.png'
        montage(path, arrays, ['GT', 'N1 / C1', 'N2 / C2', 'N3 / C3'], cell=400)
        paths['global_' + str(index)] = path
        info['subject_crop_' + str(index)] = {'box_xyxy': list(box), 'selection': 'GT RGB max>30 bounds plus25px; shared crop; resized for print only'}
    index, box = 38, (620, 350, 735, 465)
    gt = rgb(Path(models[4]['directory']) / 'test/ours_50000/gt/00038.png')
    x0, y0, x1, y1 = box
    target = gt[y0:y1, x0:x1]
    selected = (target[:, :, 0] > 100) & (target[:, :, 1] > 90) & (target[:, :, 2] < 130)
    arrays, names, color_rows = [target], ['GT'], []
    for m in [models[1], *models[4:]]:
        crop = rgb(Path(m['directory']) / 'test/ours_50000/renders/00038.png')[y0:y1, x0:x1]
        pixels = crop[selected]
        color_rows.append({'id': m['id'], 'mean_rgb': (pixels.mean(0) / 255).tolist(), 'red_zero_fraction': float((pixels[:, 0] == 0).mean())})
        arrays.append(crop); names.append(m['id'])
    paths['color_roi'] = montage(ASSETS / 'frame153_color_roi.png', arrays, names, cell=280)
    info['color_roi'] = {'original_frame': 153, 'box_xyxy': list(box), 'selection': 'GT R>100,G>90,B<130, same pixels for every model',
                         'pixel_count': int(selected.sum()), 'gt_mean_rgb': (target[selected].mean(0) / 255).tolist(), 'models': color_rows}
    im = rgb(Path(models[6]['directory']) / 'test/ours_50000/renders/00038.png') / 255.
    g = gt / 255.
    blue = (g[:, :, 2] > .3) & (g[:, :, 2] > 1.3 * g[:, :, 0]) & (g[:, :, 2] > 1.3 * g[:, :, 1])
    bad = blue & (im.mean(2) < .08) & (g.mean(2) > .18)
    components, count = label(bad, structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    component_id = int(sizes.argmax()); ys, xs = find_objects(components)[component_id - 1]
    x0, x1 = max(0, xs.start - 18), min(1600, xs.stop + 18)
    y0, y1 = max(0, ys.start - 18), min(900, ys.stop + 18)
    arrays = [gt[y0:y1, x0:x1]] + [rgb(Path(m['directory']) / 'test/ours_50000/renders/00038.png')[y0:y1, x0:x1] for m in models[4:]]
    paths['defect_roi'] = montage(ASSETS / 'frame153_blue_defect.png', arrays, ['GT', 'N1', 'N2', 'N3'], cell=300)
    with Image.open(track(Path(models[6]['directory']) / 'test/ours_50000/alpha/00038.png')) as a:
        alpha = np.array(a.convert('L')) / 255.
    pixels = components == component_id
    info['defect_roi'] = {'original_frame': 153, 'box_xyxy': [x0, y0, x1, y1], 'pixel_count': int(pixels.sum()),
                          'selection': 'Posthoc largest8-connected C3 dark-blue-error component: GT B>.3,B>1.3R,B>1.3G,GT mean>.18,pred mean<.08',
                          'mean_predicted_alpha': float(alpha[pixels].mean()),
                          'fraction_alpha_below_half': float((alpha[pixels] < .5).mean())}
    return paths, info


class Page(L.Page):
    def __init__(self, number, title, subtitle=''):
        RECORD.extend(['', '## ' + title, '', subtitle])
        super().__init__(number, '修正与模块消融 / C1-C3', title, subtitle)

    def text(self, text, style='body', gap=9):
        RECORD.append(unescape(re.sub('<[^>]+>', '', text.replace('<br/>', '\n'))))
        super().text(text, style, gap)

    def heading(self, text):
        RECORD.extend(['', '### ' + text]); super().heading(text)

    def callout(self, text):
        RECORD.append(unescape(re.sub('<[^>]+>', '', text.replace('<br/>', '\n'))))
        super().callout(text)

    def table(self, data, widths, **kwargs):
        RECORD.append('| ' + ' | '.join(map(str, data[0])) + ' |')
        RECORD.append('| ' + ' | '.join(['---'] * len(data[0])) + ' |')
        RECORD.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in data[1:])
        super().table(data, widths, **kwargs)

    def image(self, path, height=None, caption=None):
        if height is None:
            with Image.open(path) as im:
                height = L.CW * im.height / im.width
        RECORD.append(f'![{caption or Path(path).stem}](color_repair_assets/{Path(path).name})')
        super().image(path, height, caption)

    def path(self, path):
        self.text(L.path_markup(str(path.resolve())), 'path', gap=5)


def build(models, protocol, comparisons, figures, cases):
    for name, filename in [('CN', 'msyh.ttc'), ('CN-Bold', 'msyhbd.ttc')]:
        pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + filename, subfontIndex=0))
    pdfmetrics.registerFontFamily('CN', normal='CN', bold='CN-Bold', italic='CN', boldItalic='CN-Bold')
    for name, filename in [('Arial', 'arial.ttf'), ('PathFont', 'consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + filename))
    L.TOTAL_PAGES = PAGES; L.page_records.clear()
    L.c = canvas.Canvas(str(PDF), pagesize=(L.W, L.H), pageCompression=1)
    L.c.setTitle('PIDG C1-C3 修正与模块对比报告')
    L.c.setAuthor('PhysGauss / PIDG Extension Project')
    cw = L.CW
    new = models[4:]
    p = Page(1, 'C1-C3 修正与模块对比报告', 'Rubber Duck | 最终50k模型 | 固定50张官方测试图 | 中文报告')
    p.callout('核心结论：新C1的四项图像指标最好；弱局部仿射约束降低了内部低覆盖率，但未提高本轮整体画质；新C3的定向增密没有显示明确的整体指标收益。')
    p.heading('最终测试结果')
    p.table([['模型', 'PSNR↑', 'SSIM↑', 'LPIPS↓', 'MS-SSIM↑']] +
            [[m['label'], f'{m["scores"]["PSNR"]:.4f}', f'{m["scores"]["SSIM"]:.6f}', f'{m["scores"]["LPIPS"]:.6f}', f'{m["scores"]["MS-SSIM"]:.6f}'] for m in new],
            [77, 100, 110, 110, cw - 397], highlight=1)
    p.text('相对对应旧版本，新C1/C2/C3的PSNR分别提高4.626、2.140、3.410 dB，四项图像指标均改善。新C1也超过C0/V3参考：PSNR +0.648 dB；这不是与原始PIDG官方全训练集基线的等价比较。')
    p.heading('这轮改变了什么')
    p.text('共同修正：平滑正值SH颜色、保持初始RGB的逆初始化、延后且减弱mask监督、加入前景均衡RGB损失。新C2增加更弱且更局部的仿射一致性；新C3再增加颜色一致性筛选与新颜色初始化的覆盖增密。')
    p.text('三组均从新的随机初始化训练，总计各50k；模型主体仍为V3的W256_D0。未加入iso、速度-轨迹导数一致性或中心约束，也未新增颜色MLP。')
    p.callout('本报告的主比较统一使用final50000，而非根据测试分数挑选快照。旧版到新版同时改变了多项设置，因此只能说明“修正组合有效”，不能把提升全部归因于某一个修正。')
    p.text('所有150张独立render、150张GT/render二联图和50张GT+C1+C2+C3四联图已生成。最后一页给出绝对保存地址。', 'small')
    p.finish()

    p = Page(2, '实验结构与公平比较', '120 train / 30 validation / 50 test；时间戳互不重叠，但仍属于交错时刻插值')
    p.table([['设置', '旧C1/C2/C3', '本轮修正版'],
             ['主体与图像目标', 'V3 W256_D0；L1、SSIM、LPIPS', '保留相同主体和V3图像目标'],
             ['mask', '权重0.1，从第1步启用', '0.03；3k后启用，2k线性ramp'],
             ['颜色与前景RGB', 'SH+0.5后硬截断；无额外前景L1', 'Softplus(beta=5)；前景L1权重0.1'],
             ['局部仿射（C2/C3）', '权重1e-4；半径0.1', '权重5e-5；半径0.05'],
             ['覆盖克隆（C3）', '每轮最多512；继承全部SH', '最多256；颜色筛选；新DC、高阶SH清零']], [96, 203, cw - 299])
    p.heading('从零训练与共享状态')
    p.callout('新的随机step0 → 共同训练至3k → 新C1继续至50k<br/>共同3k → 开启弱仿射训练至15k → 新C2继续至50k<br/>同一个完整15k状态 → 新C3继续至50k，并在18k-36k执行覆盖增密')
    p.text('3k和15k分支复制完整参数、优化器、随机状态、训练相机栈及验证历史。C2/C3的15k完整断点SHA一致，配置仅coverage_density不同；修正了旧实验在首次定向增密之前已出现分歧的混杂。CUDA分支后的非确定性仍然存在。')
    p.text('时间统一为t=(原始帧ID-1)/199；渲染与物理分支沿用V3的共同时间尺度。常规投影梯度增密保留并于15k前结束。仅120训练帧的RGB/alpha进入损失和颜色统计；30验证帧选择快照，三组选择锁定后才导出50测试帧。', 'small')
    p.text('图中N1/N2/N3表示新C1/C2/C3，O1表示旧C1。C0/V3不是未经修改的原始PIDG：它含LPIPS训练项及此前统一时间的设置。测试集具有历史调参暴露，本次划分不能把它重新变成完全未接触的新测试集。', 'small')
    p.finish()

    p = Page(3, '修正一：恢复负颜色通道的梯度', '作用于SH颜色参数化，不增加一个新的RGB预测网络')
    p.heading('旧路径与问题')
    p.callout('q_i(d) = SH_i(d) + 0.5；旧颜色 c_i = max(q_i, 0)<br/>当q_i &lt; 0时，该通道沿颜色路径的局部导数为0。<br/>新颜色 c_i = log(1 + exp(5 q_i)) / 5；dc_i/dq_i = sigmoid(5 q_i)。')
    p.text('旧模型的训练帧诊断在每组预定的9帧中发现负的原始SH颜色，前景贡献候选的负红通道比例约为旧C1 5.82%、旧C2 5.44%、旧C3 5.61%。候选ID不是精确的像素合成权重，不能由此断定每个黑斑或偏绿像素的唯一成因。')
    p.heading('具体实现')
    p.text('用形变后中心到相机的方向d=normalize(x_i(t)-camera_center)计算SH，与原CUDA路径一致，再把平滑颜色送入既有光栅器。没有修改CUDA。关闭positive_color后回到原路径，RGB与alpha通过了逐像素一致性检查。')
    p.callout('初始颜色c0保持不变：q0 = c0 + log(1-exp(-5 c0))/5<br/>DC_new = (q0 - 0.5) / 0.2820947918<br/>初始高阶SH为0；原本精确为0的RGB使用1e-4下限。')
    p.text('逆初始化只在新随机step0执行，断点恢复不重复转换。这样避免仅切换激活函数就使全部初始颜色变亮，减少不必要的起点差异。')
    p.heading('证据与边界')
    p.text('27项CPU测试和4项CUDA测试通过，包含负红通道有限差分梯度、形变后视线方向一致性、原路径渲染一致性、颜色统计的校验和恢复；三组100步短训练、真实增密和70步恢复也通过。')
    p.text('Softplus在常见有限负值附近提供可用梯度，但极负值处梯度仍可能非常小；它不保证所有像素都有正确颜色，也不解决遮挡、错误形变或缺失几何。', 'small')
    p.finish()

    p = Page(4, '修正二：协调mask与前景颜色监督', '保留原V3图像项，改变新增监督的强度、时机与像素归一化')
    p.callout('L_V3,image = 0.8 L1 + 0.2(1-SSIM) + 0.02 LPIPS<br/>r(k) = clip((k-3000)/2000, 0, 1)<br/>L_repair = r(k) [0.03 L_mask + 0.10 L_fg]<br/>L_mask = mean_p |A_render(p)-M_GT(p)|<br/>L_fg = sum_p M_GT(p) sum_c |C_c(p)-GT_c(p)| / max(3 sum_p M_GT(p), 1)')
    p.text('完整损失仍保留既有物理残差及TV等项；以上只展开图像和本轮修正项。LPIPS采用VGG，每4步计算一次并乘4作间隔补偿，输入缩放至800×450。mask和前景RGB在3k后开始，到5k达到目标权重。')
    p.heading('为什么这样修改')
    p.text('mask只要求合成alpha接近轮廓，不直接决定SH的RGB。它会更新透明度、位置、尺度和形变，间接改变遮挡与颜色混合权重；因此轮廓更正确并不意味着颜色也更正确。')
    p.text('旧版从第1步施加0.1全图alpha监督，而形变分支还处于warm-up。延后和减弱mask是为减轻早期几何与颜色的竞争；额外前景L1按前景alpha总量归一化，避免少量前景被大量黑背景稀释。')
    p.callout('对于固定几何，C(p)=sum_i w_i(p)c_i，A(p)=1-product_i(1-a_i)。<br/>mask本身没有对c_i的直接颜色梯度；RGB监督仍需经过可用的颜色参数化和正确的可见性权重。')
    p.text('本次同时改变了激活、mask权重/调度和前景RGB，尚未分别训练“只换Softplus”“只减弱mask”“只加前景L1”的对照。因此机制合理且组合指标改善，但不能给每个修正分配独立提升量。', 'small')
    p.finish()

    p = Page(5, '模块一：更弱的局部仿射一致性', '新C2 = 新C1的共同修正组合 + 弱仿射；新C3也保留此项')
    p.callout('对canonical邻域：DX_a=[X_j-X_a]，DY_a=[x_j(t)-x_a(t)]<br/>A_a = pinv(DX_a) DY_a；残差 E_a = DY_a-DX_a A_a<br/>L_aff = mean_a {mean_j ||E_a,j||² / max(mean_j ||DX_a,j||², 1e-8)}')
    p.text('基底DX、伪逆和邻域选择不求梯度，仅形变轨迹接收该正则梯度。每轮最多128锚点、每点最多8近邻，canonical距离小于0.05；至少4条有效边，且最小奇异值超过最大奇异值的1e-4才使用该邻域。')
    p.table([['参数', '旧C2/C3', '新C2/C3'], ['权重', '1e-4', '5e-5'], ['邻域半径', '0.1', '0.05'],
             ['开始/上升期', '3k / 2k', '3k / 2k'], ['计算间隔', '10步，乘10补偿', '保留'],
             ['真实物体ID隔离', '没有', '仍没有']], [150, 170, cw - 320])
    p.heading('希望获得什么，实际观察到什么')
    p.text('它允许局部一致的旋转、拉伸和剪切，并不像固定邻居距离那样把软体锁成刚体。希望抑制相邻点的非协调运动，降低内部裂缝。新C2相对新C1的内部低覆盖率下降，但PSNR降低1.081 dB，其他三项图像指标也变差。')
    p.text('候选解释：约束可能限制真实非仿射细节；多只鸭子和蓝色障碍物接触时，空间近邻也可能跨物体。缩小半径只能降低风险，不能证明实现了物体隔离。本轮没有真实ID标签或跨物体边统计，以上不是已确认的唯一原因。', 'small')
    p.finish()

    p = Page(6, '模块二：颜色一致性的定向增密', '新C3 = 新C2 + 覆盖缺口筛选与保守克隆；不是新增一个损失项')
    p.callout('d(p,t) = 1[p属于GT腐蚀前景] × 1[A(p,t)&lt;0.7] × max(0,0.95-A)<br/>S_i = sum_{t,p: i在贡献候选中} d(p,t)，至少3个不同训练时刻投票。<br/>颜色均值mu_i=sum d·RGB_GT / sum d；每通道标准差的最大值须≤0.15。')
    p.text('16k-35,999之间每20步观察一次，像素网格步长4；每像素读取最多20个贡献候选。每个有效候选累加相同缺口权重，并不是按精确遮挡贡献比例归一化。完全没有候选的像素无法提供3D克隆种子。', 'small')
    p.text('在18k、20k至36k每2k结算：按覆盖分数排序，经颜色一致性筛选后最多256点。新DC由训练RGB均值经逆Softplus初始化，高阶SH清零；保留父点，并继承轨迹索引、尺度、旋转和动态标记。opacity夹到[0.005,0.05]，每轴位置扰动标准差为0.25×min(parent scale,0.02)。', 'small')
    p.image(figures['density'], 135, '图1：本轮实际新增44、125、45、29、4、34、17、0、0、0点，合计298点。')
    p.text('新C2最终87,813点，新C3最终88,111点，仅增加约0.339%。覆盖分数、时刻投票和颜色矩作为非学习状态保存，并与完整断点校验和绑定；并未新增可学习颜色MLP或自由物理参数。', 'small')
    p.callout('观察：C3相对C2的PSNR -0.0176 dB，SSIM和MS-SSIM极小改善，LPIPS略差。不能称为稳定的整体画质提升，也不能把少量补点理解为对所有运动空洞的重建。')
    p.finish()

    p = Page(7, '修正前后：整体图像指标', '同一固定50张测试图；C0/V3参考、旧三组与新三组均为final50000')
    p.table([['模型', 'PSNR↑', 'SSIM↑', 'LPIPS↓', 'MS-SSIM↑']] +
            [[m['label'], f'{m["scores"]["PSNR"]:.4f}', f'{m["scores"]["SSIM"]:.6f}', f'{m["scores"]["LPIPS"]:.6f}', f'{m["scores"]["MS-SSIM"]:.6f}'] for m in models],
            [77, 100, 110, 110, cw - 397], highlight=5)
    p.image(figures['metric_bars'], 290, '图2：灰柱为旧版，彩柱为修正版，虚线为C0/V3。纵轴为局部范围，不能按柱形高度估算相对百分比提升。')
    p.text('对应旧版PSNR提升：新C1 +4.6262 dB，新C2 +2.1402 dB，新C3 +3.4104 dB。新C1相对C0/V3四项均更好，但内部低覆盖率等几何代理指标并非也全部占优。', 'small')
    p.finish()

    p = Page(8, '增加模块之后，变化有多大？', '先看统一final50k，再检查逐帧分布；不利用测试指标重新挑选快照')
    rows = [['比较', 'ΔPSNR', 'ΔSSIM', 'ΔLPIPS', 'ΔMS-SSIM']]
    for key, name in [('N2-N1', '仿射：新C2-新C1'), ('N3-N2', '增密：新C3-新C2')]:
        d = comparisons[key]['metric_delta']
        rows.append([name, f'{d["PSNR"]:+.4f}', f'{d["SSIM"]:+.6f}', f'{d["LPIPS"]:+.6f}', f'{d["MS-SSIM"]:+.6f}'])
    p.table(rows, [145, 75, 95, 95, cw - 410])
    p.image(figures['per_frame'], 250, '图3：上图按原始帧编号排列PSNR；下图为同帧C3-C2差值。零线以上才是增密组该帧PSNR更高。')
    c = comparisons['N3-N2']['per_frame']['PSNR']
    p.text(f'C3有{c["improved_frames"]}/50帧PSNR高于C2，逐帧差值中位数{c["median"]:+.4f} dB，但平均差为负；说明收益与退化混合，并非每一帧都变差。单次训练不能据此建立统计显著性。', 'small')
    p.text('新C1的PSNR在29/50帧高于新C2，逐帧差值中位数+0.4124 dB，平均差+1.0808 dB；平均指标占优不表示每一帧都更好。', 'small')
    p.heading('验证集选择的PSNR快照也保持相同排序')
    p.table([['模型', '验证选择iter', '该快照测试PSNR']] + [[m['label'], m['best']['PSNR']['iteration'],
             f'{read(Path(m["directory"])/"results.json")["ours_best"]["PSNR"]:.4f}'] for m in new], [140, 170, cw - 310])
    p.text('主结果始终为final50k。该补充表是预先按验证PSNR选择的快照，而非测试集上的最佳迭代。C2/C3共享同一个15k完整状态，有助于隔离增密设置；仍保留CUDA非确定性和单seed限制。', 'small')
    p.finish()

    p = Page(9, '覆盖改善与画质并不完全一致', '两类覆盖率均在2D alpha图上计算，数值越低越好；并非3D内部空洞体积')
    p.table([['模型', '前景PSNR↑', 'Mask IoU↑', '内部低覆盖%', '背景误覆盖%']] +
            [[m['label'], f'{m["coverage"]["mean"]["foreground_psnr"]:.3f}', f'{m["coverage"]["mean"]["mask_iou"]:.4f}',
              f'{100*m["coverage"]["mean"]["interior_hole_rate"]:.5f}',
              f'{100*m["alpha"]["mean"]["background_false_positive_rate"]:.3f}'] for m in models], [65, 110, 105, 120, cw - 400])
    p.callout('内部区域 I_t = {7×7邻域GT alpha最小值 &gt; 0.95}，边界外视为0。<br/>内部低覆盖% = 100 × mean_t [ |I_t ∩ {A_t&lt;0.5}| / max(|I_t|,1) ]<br/>背景 B_t = {M_t≤0.5}。<br/>背景误覆盖% = 100 × mean_t [ |B_t ∩ {A_t&gt;0.5}| / max(|B_t|,1) ]')
    p.text('GT alpha来自原始RGBA并双三次缩放至1600×900；预测alpha来自8-bit PNG。先计算每帧比例，再等权平均50帧，不是先合并全部像素。Mask IoU使用GT/预测alpha>0.5；前景PSNR仅在GT alpha>0.5区域计算。', 'small')
    p.text('新C1的内部低覆盖率0.01082%，新C2为0.00167%，新C3为0.00156%；新增仿射后明显下降，但RGB指标变差。新C1相比旧C1虽颜色更好，Mask IoU却从0.9764降至0.9161，说明较弱mask存在轮廓取舍。')
    p.text('同样，新C1的背景误覆盖从旧C1的约0.125%升至1.004%。这表明颜色修正组合改善RGB的同时，alpha对背景的排除变弱；不能把“四项图像指标均改善”扩大为“所有指标都改善”。', 'small')
    p.text('高alpha的错误深色贡献、错误遮挡或形变可在RGB里形成黑斑，却不满足A<0.5，因此不会全部计入内部低覆盖。不能只用接近0的覆盖率判定所有视觉缺陷已消失。', 'small')
    p.finish()

    p = Page(10, 'GT与三组模型的可视化对比', '从左到右：GT / 新C1 / 新C2 / 新C3；同一帧、同一GT决定的裁剪')
    p.image(figures['global_0'], 250, '图4：原始帧1。为便于印刷，按GT可见RGB边界统一裁掉大面积黑背景，并等比例缩放；不是完整原图。')
    p.image(figures['global_38'], 250, '图5：原始帧153。三组主要颜色已较接近GT，但接触/强变形附近仍有局部形状失真，C2/C3蓝色障碍物可见暗斑。')
    p.text('这两帧是早期和此前讨论过的局部问题帧，不代表50帧整体分布。完整原始四栏拼接保留每个1600×900图块，分辨率6400×950，可在最后一页的绝对路径中逐帧检查。', 'small')
    p.finish()

    p = Page(11, '颜色问题修复了多少，仍缺什么？', '第153帧的事后局部分析；裁剪和像素选择对所有模型保持一致')
    p.image(figures['color_roi'], 122, '图6：固定ROI (620,350)-(735,465)。左至右为GT、旧C1、新C1、新C2、新C3。')
    color = cases['color_roi']
    p.table([['模型', '平均R', '平均G', '平均B', 'R=0像素%'],
             ['GT', *[f'{v:.4f}' for v in color['gt_mean_rgb']], '0.00']] +
             [[row['id'], *[f'{v:.4f}' for v in row['mean_rgb']], f'{100*row["red_zero_fraction"]:.2f}'] for row in color['models']],
            [90, 100, 100, 100, cw - 390])
    p.text(f'在GT满足R>100、G>90、B<130的同一批{color["pixel_count"]}个像素上，旧C1红通道全部为0；新版三组均不再出现该现象。该观察支持本轮组合修正缓解此局部偏绿问题，不能证明所有颜色问题都已消失，也不能单独归功于Softplus。', 'small')
    p.image(figures['defect_roi'], 116, '图7：蓝色区域残留暗斑，左至右GT/N1/N2/N3。按C3最大连通暗斑事后定位，同一矩形用于全部模型。')
    defect = cases['defect_roi']
    p.text(f'该C3连通区域有{defect["pixel_count"]}个像素，预测alpha均值{defect["mean_predicted_alpha"]:.3f}；其中仅{100*defect["fraction_alpha_below_half"]:.1f}%满足alpha<0.5。暗斑并不等同于完全透明：颜色/遮挡错误与低覆盖可能并存。', 'small')
    p.text('定向增密只采样A<0.7的覆盖缺口，因此这类高alpha暗斑不会直接触发补点。这解释了该策略可能漏掉的一类错误，并不证明全部C3-C2差异都由此造成。', 'small')
    p.text('图7选择规则：GT蓝通道>0.3且高于红/绿的1.3倍，GT均值>0.18，C3预测RGB均值<0.08；取最大8连通分量并加18像素边距。该局部图仅作解释，没有反馈到本轮训练或模型选择。', 'small')
    p.finish()

    p = Page(12, '结论、后续消融与文件地址', '保留不同目标下的候选；本报告不启动任何新训练')
    p.text('<b>图像质量优先：保留新C1。</b> 本轮四项图像指标均最好，较旧版改善幅度较大；同时保留C2作为更低内部低覆盖率的候选。C3仅298点增量、收益混合，暂不设为默认整体改进。')
    p.text('<b>下一步：</b> 分别对照Softplus、mask权重/调度、前景RGB，确认组合里哪项有效；至少多个seed复验。对局部约束先验证真实邻接，特别检查跨物体接触。对增密记录新点可见贡献、颜色与局部RGB误差，而不只统计新增数量。', 'small')
    p.text('<b>结论边界：</b> 单场景、单seed、CUDA非确定性；历史测试暴露未消除；当前是交错时间插值，不是长时间缺帧或未来预测；没有GT深度、3D形变、速度或力的准确性结论。背景占比高，必须结合前景PSNR、alpha诊断与局部图查看。', 'small')
    p.heading('绝对保存地址')
    p.text('GT + 三模型四栏拼接：50张，final50k', 'small', gap=3)
    p.path(EXPORTS / 'gt_c1_c2_c3/index.html')
    p.text('独立render与GT/render二联图：各模型50张', 'small', gap=3)
    for name in NAMES:
        p.path(EXPORTS / name / 'renders')
        p.path(EXPORTS / name / 'gt_render')
    p.text('共享GT；整体浏览页面', 'small', gap=3)
    p.path(EXPORTS / 'gt')
    p.path(EXPORTS / 'index.html')
    p.text('模型权重、配置和日志位于 runs/pidg_color_repair/{C1_repaired,C2_repaired,C3_repaired}/rubber_duck。实现位于 variants/color_repair；配套assessment.json、sources.json和Markdown记录完整数值、逐帧统计及来源SHA-256。', 'small')
    p.finish()
    L.c.save()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    prior_pdf = OUT / 'PIDG_Coverage_Ablation_Report_CN.pdf'
    prior_sha = digest(prior_pdf)
    metadata, models, protocol, plan = load()
    alpha_audit(metadata, models)
    comparisons = {a['id'] + '-' + b['id']: compare(a, b) for a, b in [
        (models[4], models[1]), (models[5], models[2]), (models[6], models[3]),
        (models[4], models[0]), (models[5], models[4]), (models[6], models[5])]}
    figures = charts(models)
    visuals, cases = visual_cases(metadata, models)
    figures.update(visuals)
    build(models, protocol, comparisons, figures, cases)
    for path in [Path(__file__), ROOT / 'scripts/pidg_ablation_report_layout.py',
                 *[ROOT / 'variants/color_repair' / x for x in ('model.py', 'color.py', 'runtime.py', 'evidence.py', 'run.py', 'prepare.py')],
                 ROOT / 'variants/coverage_tuning/model.py', ROOT / 'variants/coverage_tuning/density.py',
                 ROOT / 'variants/kinematic_resplit/runtime.py', ROOT / 'variants/metric_tuning/runtime.py']:
        track(path)
    for name in ('preflight.json', 'unit_tests.json', 'original_color_diagnostic.json', 'four_panel_export.json'):
        read(REPORTS / name)
    assert digest(prior_pdf) == prior_sha
    assert len(L.page_records) == PAGES
    assessment = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol, 'plan': plan,
                  'comparisons': comparisons, 'cases': cases, 'models': models,
                  'all150_new_alpha_counts_match_saved_coverage': True,
                  'interpretation': 'Old/new is a repair-package comparison; no isolated attribution to each repair. C2/C3 share full15k, single scene/seed, GPU nondeterminism remains.'}
    (REPORTS / 'assessment.json').write_text(json.dumps(assessment, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    PDF.with_suffix('.md').write_text('# PIDG C1-C3 修正与模块对比报告\n' + '\n\n'.join(RECORD) + '\n', encoding='utf-8')
    proof = {'pdf_absolute': str(PDF), 'pdf_sha256': digest(PDF), 'expected_pages': PAGES,
             'pages': L.page_records, 'sources': SOURCES, 'protocol': protocol,
             'previous_coverage_report_unchanged_sha256': prior_sha, 'visual_case_definitions': cases,
             'figures': {str(path.relative_to(ROOT)): digest(path) for path in figures.values()}}
    PDF.with_suffix('.sources.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pdf': str(PDF), 'pages': L.page_records,
                      'new_alpha': {m['id']: m['alpha']['mean'] for m in models[4:]}, 'cases': cases}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
