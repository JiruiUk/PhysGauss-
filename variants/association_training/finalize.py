"""Compare locked final states; observed support never becomes physical ID truth."""
from pathlib import Path
import html
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from variants.metric_tuning.runtime import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/association_training'
RUNS = ROOT / 'runs/pidg_association_training'
OUT = ROOT / 'output/association_training'
NAMES = ['C1_original', 'T0_C1_continue', 'T1_association']
WINDOWS = {'early_2_27': (2, 27), 'pre_first_fragment_end_28_42': (28, 42),
           'after_first_fragment_end_43_145': (43, 145), 'all_early_IDs_unverified_146_200': (146, 200)}


def observed_flow_audit(name):
    """Raw FB-visible motion agreement, conditional on center+visible footprint."""
    from variants.flow_comparison.geometry import forward_backward_valid, sample
    from scipy.ndimage import binary_erosion
    z = np.load(RUNS / 'evaluation_cache' / name / 'evidence.npz')
    frames, means, mass = z['frames'], z['means'], z['mass']
    with np.load(ROOT / 'output/surface_support/image_tracks.npz') as tracks:
        components = dict(zip(tracks['frames'], tracks['components']))
    manifest = read_json(ROOT / 'data/derived/flow_comparison_train_only/manifest.json')
    rows = []
    for i, pair in enumerate(manifest['pairs']):
        assert [int(frames[i]), int(frames[i + 1])] == [pair['source'], pair['target']]
        with np.load(ROOT / 'data/derived/flow_comparison_train_only' / pair['file']) as flow:
            f = torch.tensor(flow['forward'].astype(np.float32))
            b = torch.tensor(flow['backward'].astype(np.float32))
        xy = torch.tensor(means[i])[:, None, :]
        valid = (sample(forward_backward_valid(f, b).float(), xy)[:, 0] > .99).numpy()
        error = torch.linalg.vector_norm(torch.tensor(means[i + 1] - means[i]) - sample(f, xy)[:, 0], dim=-1).numpy()
        for offset in [0, 1]:
            pts = np.rint(means[i + offset]).astype(np.int64)
            inside = (z['depth'][i + offset] > .2) & (pts[:, 0] >= 0) & (pts[:, 0] < 400) & (pts[:, 1] >= 0) & (pts[:, 1] < 225)
            mask = binary_erosion(components[int(frames[i + offset])] >= 0, iterations=1)
            positions = np.flatnonzero(inside)
            inside[positions] &= mask[pts[positions, 1], pts[positions, 0]]
            inside &= (mass[i + offset, :, 0] >= .02) & (mass[i + offset, :, 0] / np.maximum(mass[i + offset, :, 1], 1e-20) >= .95)
            valid &= inside
        count = int(valid.sum())
        rows.append({'source': int(frames[i]), 'target': int(frames[i + 1]), 'observable_points': count,
                     'points_with_error_le_2px': int((valid & (error <= 2)).sum()),
                     'mean_error_pixels': float(error[valid].mean()) if count else None,
                     'note': 'Any-duck visible support and motion only; no cross-occlusion ID inference.'})
    write_json(REPORT / (name + '_observed_flow.json'), rows)
    return rows


def main():
    torch.set_num_threads(4)
    assert read_json(REPORT / 'verification.json')['passed']
    support = {n: read_json(REPORT / (n + '_support.json')) for n in NAMES}
    valid = {n: read_json(REPORT / (n + '_validation.json'))['metrics'] for n in NAMES}
    flow = {n: observed_flow_audit(n) for n in NAMES}
    summary = {}
    for name in NAMES:
        rows = support[name]['frames']
        windows = {}
        for key, (lo, hi) in WINDOWS.items():
            rr = [r for r in rows if lo <= r['frame'] <= hi]
            ff = [r for r in flow[name] if lo <= r['target'] <= hi]
            observable = sum(r['observable_points'] for r in ff)
            windows[key] = {'frames': len(rr),
                'mean_early_bank_alpha_share_on_any_duck': float(np.mean([r['early_bank_alpha_share_on_any_duck'] for r in rr])),
                'mean_bank_mass_outside_ducks': float(np.mean([r['fraction_bank_mass_outside_ducks'] for r in rr])),
                'mean_supported_points_on_verified_IDs': float(np.mean([r['supported_points_on_independently_verified_IDs'] for r in rr])),
                'frames_with_four_supported_candidate_groups': sum(bool(r['candidate'] and r['candidate']['all_four_groups_supported']) for r in rr),
                'observable_point_pairs': observable,
                'observed_flow_agreement_fraction': sum(r['points_with_error_le_2px'] for r in ff) / observable if observable else None}
        summary[name] = {'validation': valid[name], 'windows': windows,
                         'selected_frames': [r for r in rows if r['frame'] in [2, 27, 43, 55, 63, 74, 86, 108, 134, 172, 188, 200]]}
    delta = {k: valid[NAMES[2]][k] - valid[NAMES[1]][k] for k in valid[NAMES[1]]}
    loss_events = [json.loads(s) for s in (RUNS / 'pilot/T1_association/rubber_duck/association_loss.jsonl').read_text().splitlines()]
    loss_windows = {}
    for key, (lo, hi) in WINDOWS.items():
        events = [r for r in loss_events if lo <= r['pair'][0] <= hi]
        loss_windows[key] = {'events': len(events), 'zero_temporal_events': sum(r['temporal'] == 0 for r in events),
                             'mean_temporal': float(np.mean([r['temporal'] for r in events])) if events else None}
    result = {'status': 'completed', 'models': summary, 'T1_minus_T0_validation': delta,
              'temporal_training_signal_by_window': loss_windows,
              'true_physical_ID_accuracy_available': False, 'physical_reidentification_demonstrated': False,
              'conclusion': 'Early local purity improves while visible contribution shrinks; no post-occlusion support extension in this matched2000-update pilot.',
              'limits': 'One fixed-weight continuation pilot, one scene and seed; not proof that all association losses or longer/different training must fail.',
              'window_definition': WINDOWS, 'manual_review': 'See manual_review.json; generated images do not imply they were all manually inspected.'}
    write_json(REPORT / 'results.json', result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for name in NAMES:
        rr = support[name]['frames']
        axes[0].plot([r['frame'] for r in rr], [100 * r['early_bank_alpha_share_on_any_duck'] for r in rr], label=name)
        axes[1].plot([r['frame'] for r in rr], [100 * r['fraction_bank_mass_outside_ducks'] for r in rr], label=name)
    axes[0].set_ylabel('Early-bank share of duck alpha (%)')
    axes[1].set_ylabel('Bank mass outside ducks (%)'); axes[1].set_xlabel('Original training frame')
    for ax in axes:
        ax.axvline(43, color='gray', linestyle='--', alpha=.5)
        ax.axvline(146, color='gray', linestyle=':', alpha=.7)
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Visible support of the same 2,065 early Gaussian rows; not physical ID accuracy')
    fig.tight_layout(); fig.savefig(OUT / 'support_comparison.png', dpi=160); plt.close(fig)
    early0 = summary[NAMES[1]]['windows']['early_2_27']
    early1 = summary[NAMES[2]]['windows']['early_2_27']
    lines = ['# 关联链训练试验结果', '',
             '**结果：早期局部关联更干净，但本轮未改善遮挡后的固定点支持或身份恢复。** 原始C1、普通续训和关联续训均在原始帧82及之后的全部已评估训练时刻失去早期点库对鸭子区域的贡献，未延长支持时段。此结论限于本轮固定权重、单场景/种子的2,000步续训，不说明所有关联监督都无效。', '',
             '同一 C1 50k 完整模型、优化器、RNG 和相机栈各续训 2,000 次更新，终点 52k。T0 关闭新约束；T1 使用早期已核验点库锚点和原始光流可见连续性。训练为1600×900，点数均86,143，固定身份锚点2,065个。', '',
             '## 固定最终完整状态的验证集重建', '',
             '三组使用相同30张验证图像和PNG量化。LPIPS采用VGG与标准[-1,1]输入；历史训练日志将[0,1]直接送入LPIPS，因此其LPIPS数值不能与本表直接混用。', '',
             '| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | MS-SSIM↑ |', '|---|---:|---:|---:|---:|']
    for name in NAMES:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{valid[name][k]:.7f}' for k in ['PSNR', 'SSIM', 'LPIPS', 'MS-SSIM']) + ' |')
    lines += ['', '## 点库可见支持', '',
              '下表为早期固定点库贡献占全部鸭子区域alpha的比例；这是表面覆盖支持量，不是身份准确率，也不是要求一个局部表面覆盖整只鸭子。', '',
              '| 时间窗口 | C1原始 | T0续训 | T1关联 |', '|---|---:|---:|---:|']
    for window in WINDOWS:
        values = [summary[n]['windows'][window]['mean_early_bank_alpha_share_on_any_duck'] for n in NAMES]
        lines.append('| ' + window + ' | ' + ' | '.join(f'{v:.4%}' for v in values) + ' |')
    lines += ['',
              f"早期2–27，点库贡献落到鸭子区域外的比例由T0的{early0['mean_bank_mass_outside_ducks']:.4%}降到T1的{early1['mean_bank_mass_outside_ducks']:.4%}；中心与可见足迹同时支持既有独立ID的平均点数由{early0['mean_supported_points_on_verified_IDs']:.1f}升到{early1['mean_supported_points_on_verified_IDs']:.1f}。局部关联纯度有改善，但总可见贡献由75.14%降到65.22%。", '',
              '原始帧43之后，三组都没有任何一帧通过“四组同时有完整足迹与中心支持”的候选门槛。原74帧点库alpha份额T0为20.87%，T1为15.07%；原86帧起已无鸭子区域贡献。', '',
              '## 有证据的失效机制与限制', '',
              '晚期146–200的29个训练帧对，时序关联项29次全部为零；全部原始FB可见点对也没有符合两端中心与鸭子足迹条件的旧库点。旧库仍可能在蓝色结构或背景位置产生贡献，不能把“有点/有颜色”算作鸭子身份。', '',
              '约束会减少早期错误区域贡献，同时80%单侧参考下限允许一定可见贡献减少。这与“局部纯度提高、贡献缩小”的结果相符；没有做参数冻结分解，不能把贡献减少唯一归因于opacity或尺度。晚期旧库在可监督鸭子区域缺少有效信号，二维可见光流本身也未提供被遮挡期间的独立身份桥。', '',
              '43–145窗口内，可观测点对的流误差≤2px比例T0为95.86%、T1为96.28%，对应点对数14,208/14,116。该条件性小幅改善没有伴随长时支持恢复。', '',
              '建议保留C1主线，下一步优先建立可独立核验的跨遮挡证据，并为持续物体/表面状态制定单独开关和对照试验。当前结果不支持直接用此分支替换基线，未启动新一轮训练或恢复旧P1/O0。']
    lines += ['', '## 身份判断的边界', '',
              '独立图像片段在原始帧43开始有身份终止，帧146后早期四个身份全部不可核验。终止后不复活，也不把晚期新片段强制并入早期ID。模型最优排列及其间隔只表示候选兼容性；没有独立真实物体ID真值，不能将其称为重识别准确率。', '',
              '当前目标仅作用于有可见贡献的固定早期表面点库。完全不可见或后期由另一批Gaussian解释的表面缺少跨遮挡监督；本试验不增加持续物体状态、接触约束、形状损失或材料模型。观察光流指标只统计两端中心、完整可见足迹和原始FB均有效的点对，不把未观测点记作成功。', '',
              '训练与输入哈希、相同2,000次相机采样、优化器步数、360帧完整贡献积分以及历史缓存比对见 `verification.json`。', '',
              '人工审阅：支持曲线、RGB/点库贡献/投影中心三张接触表（原43/74/86/200，各三模型），以及T1原74/200的1600×900 GT|Render；未逐张人工审阅其余全部观测。', '',
              '[比较图和逐帧页面](../../output/association_training/index.html)', '']
    (REPORT / 'assessment.md').write_text('\n'.join(lines), encoding='utf-8')
    for kind, title in [('rgb', 'RGB reconstruction'), ('groups', 'Fixed early-bank visible contribution'), ('centers', 'Fixed early-bank projected centers')]:
        panel = Image.new('RGB', (1200, 4 * 249 + 30), '#17202b')
        draw = ImageDraw.Draw(panel)
        for k, name in enumerate(NAMES): draw.text((k * 400 + 8, 7), name, fill='white')
        for r, number in enumerate([43, 74, 86, 200]):
            y = 30 + r * 249
            draw.text((8, y + 4), f'Frame {number} / {title}', fill='white')
            for k, name in enumerate(NAMES):
                panel.paste(Image.open(OUT / name / f'{kind}{number:04d}.png'), (k * 400, y + 24))
        panel.save(OUT / (kind + '_comparison.png'))
    gallery = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>关联训练对照</title>',
               '<style>body{font:16px system-ui;background:#16202d;color:#eee;margin:24px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px}a{color:#7dd3fc}</style>',
               '<h1>关联训练：固定最终完整状态</h1><p>列：原始C1 / 同预算续训T0 / 关联约束T1。彩色为早期固定点库的可见贡献，黑色区域表示这些点没有显著贡献，不等于物体不存在。</p>',
               '<p>各颜色只表示早期局部点库；候选排列不能确认遮挡后真实身份。</p>']
    gallery.append('<a href="support_comparison.png"><img src="support_comparison.png"></a>')
    for kind in ['rgb', 'groups', 'centers']:
        gallery.append(f'<h2>{kind}</h2><a href="{kind}_comparison.png"><img src="{kind}_comparison.png"></a>')
    for number in [2, 14, 27, 32, 42, 43, 55, 63, 74, 86, 108, 134, 172, 188, 200]:
        gallery.append(f'<h2>Frame {number}</h2><table><tr>' + ''.join('<th>' + n + '</th>' for n in NAMES) + '</tr>')
        for kind in ['rgb', 'groups', 'centers']:
            gallery.append('<tr>' + ''.join(f'<td><img src="{n}/{kind}{number:04d}.png"></td>' for n in NAMES) + '</tr>')
        gallery.append('</table>')
    (OUT / 'index.html').write_text('\n'.join(gallery), encoding='utf-8')
    print(json.dumps({'validation': valid, 'delta': delta, 'window_support': {n: summary[n]['windows'] for n in NAMES}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
