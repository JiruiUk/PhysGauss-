"""Finalize measured pilot only after separate actual visual review."""
from pathlib import Path
import json
import msvcrt
import os
import sys
import numpy as np
import psutil
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from variants.staged_reconstruction.prepare_data import read, write, sha
from variants.staged_reconstruction.cross_temporal_static.prepare import REPORT, RUNS, OUT, verify


def main():
    verify()
    result = read(REPORT/'exports.json'); review = read(REPORT/'manual_review.json')
    repair = read(REPORT/'evaluation_repair_integrity.json')
    assert repair['passed'] and repair['training_weights_and_protocol_unchanged']
    test_ids = {int(Path(f['file_path']).stem) for f in read(ROOT/'data/derived/kinematic_resplit/transforms_test.json')['frames']}
    assert all(r['width'] == 400 and int(Path(r['path']).stem) in test_ids for r in repair['original_dataset_RGB_fallback'])
    assert review['native_images_actually_viewed']
    assert review['checkpoint_sha256'] == result['checkpoint_sha256']
    for path, expected in review['files_reviewed'].items(): assert sha(ROOT/path) == expected
    active = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.pid != os.getpid() and 'python' in (p.info['name'] or '').lower():
            if any(a == 'variants.staged_reconstruction.cross_temporal_static.train' or
                   a == 'variants.staged_reconstruction.cross_temporal_static.evaluate'
                   for a in p.info['cmdline'] or []): active.append(p.pid)
    assert not active, active
    with (RUNS/'training.lock').open('r+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    audits = {arm: read(REPORT/f'formal_{arm}_audit.json') for arm in ['fixed', 'progressive']}
    assert all(a['passed'] and a['iteration'] == 12000 for a in audits.values())
    assert result['matched_initial_model_and_camera_sequence'] and result['video']['decoded_frames'] == 200
    from plyfile import PlyData
    import torch
    for arm, info in result['PLY'].items():
        vertices = PlyData.read(ROOT/info['path'])['vertex'].data
        state = torch.load(RUNS/'formal'/arm/'checkpoint.pt', map_location='cpu', weights_only=False)
        xyz = np.stack([vertices[name] for name in ['x', 'y', 'z']], 1)
        assert len(vertices) == 1954 and np.array_equal(xyz, state['model']['_xyz'].numpy())
        assert all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
    events = result['confidence_history']; initial = result['confidence_summary']['fixed']['fraction_all_image_pixels']
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    steps = [0]+[e['step'] for e in events]
    ax[0].plot(steps, [initial*100]+[e['fraction_all_image_pixels']*100 for e in events], 'o-', label='Progressive')
    ax[0].axhline(initial*100, color='#cc8844', label='Fixed')
    cap = read(REPORT/'discovery.json')['candidate_upper_bound']['fraction_all_image_pixels']*100
    ax[0].axhline(cap, color='gray', linestyle=':', label='Geometric candidate cap')
    ax[0].set(xlabel='Actual optimizer updates', ylabel='Selected pixels / all train pixels (%)', title='Coverage proxy, not static recall')
    ax[0].legend(fontsize=8)
    ax[1].bar([e['step']-150 for e in events], [e['newly_admitted'] for e in events], width=300, label='Admitted')
    ax[1].bar([e['step']+150 for e in events], [-e['revoked'] for e in events], width=300, label='Revoked')
    ax[1].set(xlabel='Actual optimizer updates', ylabel='400px image pixels per update', title='Actual confidence threshold crossings')
    ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/'confidence_history.png'); plt.close(fig)
    lines = ['# S1-CT：跨时间置信度静态重建对照', '', review['verdict'], '',
        '只使用120张训练RGB、时间和数据相机；alpha仅合成RGB。没有静态颜色标签、旧模型权重、物体数或材料/接触/未知力。借鉴论文的置信度更新机制，但没有其完整历史3DGS或DUSt3R先验，因此不是论文复现。', '',
        '## 实验与可信度', '',
        '53个去重几何锚点（14 SIFT、39 LK），两组共同初始化1954高斯，各12000实际Adam更新、相同相机序列、同一优化器。fixed固定初始置信度；progressive每1000步更新，几何候选为上限，RGB误差可抑制或撤销监督。未知区域不是背景。', '',
        '与旧S1-A相比，宽基线匹配、去重/时间条件、候选初始化和初始opacity均有变化；不能把新旧模型差异全归于置信度。fixed与progressive配对比较才隔离本次置信度更新。', '',
        '| 共同评估区域 | 模型 | 非空图数 | 区域PSNR(dB) | 区域alpha均值 | alpha<0.5比例 |',
        '|---|---|---:|---:|---:|---:|']
    labels = {'common_S1_A_train_proxy': '冻结旧S1-A训练区域', 'common_candidate_train_proxy': '共同候选训练区域',
              'val_inferred_geometric_region_not_GT': '独立验证相机上的推断区域'}
    for label, arms in result['metrics'].items():
        for arm, m in arms.items():
            values = [f'{m[k]:.4f}' if m[k] is not None else '无支持' for k in ['PSNR', 'alpha_mean', 'alpha_below_05']]
            lines.append(f'| {labels[label]} | {arm} | {m["nonempty_frames"]} | '+ ' | '.join(values)+' |')
    lines += ['', '以上PSNR是各非空图像区域PSNR的均值。训练区域是选择条件下的代理量；验证图像未进入训练，但其区域同样由几何与RGB推断，不是独立静态标签或几何GT。验证区域筛选和相机覆盖仍可能偏向容易区域。', '',
        '## 监督扩展与限制', '']
    for arm, c in result['confidence_summary'].items():
        lines.append(f'- {arm}：{c["positive_pixels"]}个400px正支持像素，{c["frames_any"]}/120图非空，{c["frames_64"]}图至少64像素；占所有训练像素{100*c["fraction_all_image_pixels"]:.4f}%。有非零图像监督的优化步数{audits[arm]["nonzero_supervision_steps"]}/12000。')
    lines += ['', '分母包含背景和动态内容，上述比例不能视为静态召回率。新增/撤销次数可重复计入同一像素，不能当成独立发现数量。两组相机池相同，fixed在原无支持图像的步骤没有图像梯度，只有共同位置正则和Adam状态演化；12000总更新不等于12000有效图像监督。', '',
        '## 实际图像复核', '']
    lines += ['- '+finding for finding in review['findings']]
    lines += ['', '## 检查与产物', '',
        '29项测试、两组独立进程40→80恢复（含置信度历史、Adam、全部RNG、相机序列）、原生GaussianModel精确渲染对照、正式12k状态审计、共同初始化/采样对照通过。冻结源码/输入及历史S1-A/S1–S6/S3-U检查点匹配。原轨迹H264 Main/yuv420p/faststart对照视频完整解码200帧；实际人工查看记录限于图片，不声称看完整视频或浏览器播放。', '',
        '首轮预检暴露置信度公式在实际上限下无法撤销；保留原源码、反例和预检状态，正式训练前修正并增加实际上限测试。还保留pytest临时目录及CUDA环境激活错误。未根据正式结果扫阈值/损失权重。', '',
        '首轮视频导出误从derived训练目录读取完整轨迹的测试RGB而失败；冻结评估源和不完整输出保留。独立evaluate_repair.py仅为缺失的测试图像读取原始数据RGB，最新完整结果写入evaluation_v2；核实fallback只发生在400px视频观察图、属于测试split，没有改变训练/验证数值目标、权重、协议或指标门槛。PLY逐顶点位置与检查点完全一致且所有字段有限。', '',
        '[图像/视频/PLY](../../../output/staged_reconstruction/S1_cross_temporal/index.html) · [完整数值](exports.json) · [原始图像复核](manual_review.json)', '',
        '## 计划修订', '', review['next_step'], '',
        '本次只完成独立静态重建对照，不替换S3-U静态池或续训其动态模型。无完整静态几何GT时，不以覆盖增加、零时变误差或局部PSNR代替几何完整性证据。']
    (REPORT/'assessment.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    completion = {'experiment_completed': True, 'full_static_reconstruction_success': review['full_static_reconstruction_success'],
        'checkpoint_sha256': result['checkpoint_sha256'], 'matched_initialization_and_camera_sequence': True,
        'formal_updates': {a: 12000 for a in audits}, 'nonzero_supervision_steps': {a: audits[a]['nonzero_supervision_steps'] for a in audits},
        'tests_passed': 29, 'source_manifest_files_verified': len(read(REPORT/'source_freeze.json')),
        'historical_models_preserved': True, 'training_lock_free': True, 'active_trainers_or_evaluators': active,
        'static_bank_replacement_admitted': False, 'independent_static_geometry_GT': False,
        'video_decoded_frames': 200, 'manual_review_sha256': sha(REPORT/'manual_review.json'),
        'PLY_all_vertices_equal_checkpoint_and_finite': True, 'video_RGB_fallback_only_test_split_and_400px': True,
        'assessment_sha256': sha(REPORT/'assessment.md'), 'confidence_chart_sha256': sha(OUT/'confidence_history.png')}
    write(REPORT/'completion_integrity.json', completion)
    write(RUNS/'state.json', {'status': 'completed_reviewed', 'stage': 'S1-CT', 'updates_per_arm': 12000,
        'full_static_reconstruction_success': review['full_static_reconstruction_success'],
        'replacement_admitted': False, 'checkpoint_sha256': result['checkpoint_sha256']})
    print('CROSS_TEMPORAL_FINALIZED', completion, flush=True)


if __name__ == '__main__': main()
