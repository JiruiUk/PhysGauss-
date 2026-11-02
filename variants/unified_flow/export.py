"""Same pixel-verified artifacts, plus separate controlled follow-up results."""
from pathlib import Path
import torch
from variants.flow_comparison import export as old
from variants.metric_tuning.run import read_json as read, write_json as write
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT/'experiments/unified_flow'; RUNS = ROOT/'runs/pidg_unified_flow'
OUT = ROOT/'output/unified_flow'
NAMES = ['L0_corrected_trajectory', 'L1_shared_stress']


def export(directory):
    old.REPORTS, old.RUNS, old.OUT = REPORTS, RUNS, OUT
    old.export(directory)


def report():
    from .evaluate import evaluate
    motion = evaluate()
    rows = [{'model': n, 'snapshot': tag, **read(RUNS/n/'rubber_duck/results.json')['ours_'+tag]}
            for n in NAMES for tag in ['best', '50000']]
    selected = read(RUNS/'selection_lock.json')['models']
    verification = {}
    prefix = torch.load(RUNS/'shared_prefix/rubber_duck/checkpoints/resume.pt', map_location='cpu', weights_only=False)
    for n in NAMES:
        run = RUNS/n/'rubber_duck'
        state = torch.load(run/'checkpoints/resume.pt', map_location='cpu', weights_only=False)
        assert state['iteration'] == 50000
        assert all(torch.isfinite(v).all() for key in ['deform', 'grid4d', 'velocity_net', 'gaussian_parameters'] for v in state[key].values())
        if n == NAMES[0]:
            assert all(torch.equal(state['deform'][k], prefix['deform'][k]) for k in prefix['deform'] if k.startswith('shared_stress.'))
        else:
            assert all(torch.equal(state['velocity_net'][k], prefix['velocity_net'][k]) for k in prefix['velocity_net'])
            assert not torch.equal(state['deform']['shared_stress.weight'], prefix['deform']['shared_stress.weight'])
        verification[n] = {'full_iteration': 50000, 'finite': True, 'unused_heads_unchanged': True}
    write(REPORTS/'results.json', {'rows': rows, 'selection': selected, 'motion': motion, 'verification': verification, 'protocol': read(REPORTS/'plan.json')})
    lines = ['# 修正光流与共享应力分支对照', '',
        '两组从相同扩展完整3k状态到总50k；只改变应力来源。L0保留独立物理应力场；L1使用形变Grid4D及同一latent的应力头。速度都由完整轨迹导出。',
        '两组都使用修正的raw source→target观测光流，继承实际加载器相机约定、修正offset符号及混合权重。相对K2还共同改变了流程，不将这部分算作共享分支的收益。', '',
        '| 模型 | 快照 | 测试 PSNR | SSIM | LPIPS | MS-SSIM |', '|---|---|---:|---:|---:|---:|']
    for r in rows: lines.append('| '+r['model']+' | '+r['snapshot']+' | '+' | '.join(f'{r[k]:.7f}' for k in ['PSNR', 'SSIM', 'LPIPS', 'MS-SSIM'])+' |')
    lines += ['', '| 模型 | 验证 PSNR | 轨迹 EPE px | 速度推进 EPE px |', '|---|---:|---:|---:|']
    for r in motion['results']:
        if r['snapshot'] == 'best':
            n = r['model']; m = r['metrics']
            lines.append(f"| {n} | {selected[n]['value']:.5f} | {m['geometric_EPE_px']:.5f} | {m['velocity_EPE_px']:.5f} |")
    lines += ['', '单场景单种子、历史测试暴露；二维流为估计量，无物理三维速度/应力真值；共享latent不代表本构正确，也不保证解决二阶时间差分的尺度问题。没有外力或材料未知量。', '']
    for n in NAMES:
        lines += [f'- [{n} GT|Render]({(OUT/n/"index.html").as_posix()})', f'- [{n} 视频]({(OUT/n/"GT_Render_best.mp4").as_posix()})']
    (REPORTS/'assessment.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
