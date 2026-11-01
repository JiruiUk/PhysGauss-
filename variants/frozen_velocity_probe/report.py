"""Read completed diagnostic artifacts; produce scientific plots and a CN report."""
import importlib.util
import json
import os
from pathlib import Path
import statistics
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'experiments/frozen_velocity_probe'
OUT=ROOT/'runs/pidg_frozen_velocity_probe/H3_final_20261101'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    torch.set_num_threads(4)
    from .core import alignment_metrics
    result=read(OUT/'results.json');manifest=read(OUT/'manifest.json')
    control_path=REPORT/'clock_control_result.json'
    if control_path.exists():
        control=read(control_path)
        if control['arm']['name'].startswith('C1_'):result['arms'].append(control)
    data=torch.load(OUT/'targets.pt',map_location='cpu',weights_only=True)
    spec=importlib.util.spec_from_file_location('probe_report_pidg',ROOT/'variants/integrity_tuning/code/motion_utils/pidg.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=module.VelocityNet();source.load_state_dict(data['source_velocity']);source.eval()
    ts=torch.linspace(0,1,1001)[:,None]
    def latent(net):
        with torch.no_grad():values=torch.tanh(net.time_proj(net.time_embed(ts))).reshape(-1)
        increments=values.diff();nonzero=increments[increments.abs()>1e-7]
        return dict(min=values.min().item(),max=values.max().item(),
          turning_points=int((nonzero[1:]*nonzero[:-1]<0).sum()),
          positive_fraction=(increments>1e-7).float().mean().item(),
          negative_fraction=(increments<-1e-7).float().mean().item(),values=values.tolist())
    latent_data={'source':latent(source)};time_metrics={};particle_metrics={};checkpoint_checks={}
    for arm in result['arms']:
        name=arm['arm']['name']
        checkpoint=torch.load(OUT/name/'velocity_only.pt',map_location='cpu',weights_only=False)
        net=module.VelocityNet();net.load_state_dict(checkpoint['velocity_net']);net.eval()
        if checkpoint.get('clock_mode'):
            from .clock import apply_clock
            apply_clock(net,checkpoint['clock_mode'])
        latent_data[name]=latent(net)
        metrics=[]
        with torch.no_grad():
            # Full fitting-particle population at each of the30 validation times.
            x=data['X'][:data['ntrain']];ids=data['particle_indices'][:data['ntrain']]
            for i,tv in enumerate(data['validation_times']):
                t=x.new_full((len(x),1),tv.item())
                pred=net(torch.cat([x,t],-1),ids)[0]
                metrics.append(dict(time=tv.item(),**alignment_metrics(pred,data['targets']['validation'][i,:len(x)],result['movement_threshold'])))
            train_predictions=[]
            for tv in data['train_times']:
                t=x.new_full((len(x),1),tv.item())
                train_predictions.append(net(torch.cat([x,t],-1),ids)[0])
            tp=torch.stack(train_predictions);td=data['targets']['train'][:,:len(x)]
            per_point=((tp-td).square().mean((0,2)).sqrt()/td.square().mean((0,2)).sqrt().clamp_min(1e-8))
            particle_metrics[name]={f'point_relative_rmse_q{q}':torch.quantile(per_point,q/100).item() for q in [25,50,75,90,95]}
        time_metrics[name]=metrics
        relative=(sum(m['rmse']**2*m['count'] for m in metrics)/sum(m['target_rms']**2*m['count'] for m in metrics))**.5
        expected=arm['final']['metrics']['known_validation']['relative_rmse']
        assert abs(relative-expected)<1e-5,(name,relative,expected)
        checkpoint_checks[name]={'saved_checkpoint_cpu_relative_rmse':relative,'logged_gpu_relative_rmse':expected,'absolute_difference':abs(relative-expected)}
    (REPORT/'independent_checkpoint_check.json').write_text(json.dumps(checkpoint_checks,indent=2),encoding='utf8')
    energy=data['targets']['train'][:,:data['ntrain']].square().sum(-1).reshape(-1)
    top1=energy.topk(max(1,len(energy)//100)).values.sum()/energy.sum()
    (REPORT/'temporal_details.json').write_text(json.dumps(dict(latent=latent_data,per_validation_time=time_metrics,
      per_training_particle=particle_metrics,training_target_top1pct_vector_energy_fraction=top1.item()),ensure_ascii=False,indent=2),encoding='utf8')

    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl-frozen-velocity-probe'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties,fontManager
    fontManager.addfont('C:/Windows/Fonts/Deng.ttf')
    font=FontProperties(fname='C:/Windows/Fonts/Deng.ttf')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':9.5})
    colors=['#1b678d','#d47d30','#389777','#9165a5']
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    histories={}
    labels={'A_small_overfit':'A：小样本','B1_pretrained_lr1e-3':'B1：原权重 / 1e-3','B2_pretrained_lr3e-4':'B2：原权重 / 3e-4','B3_scratch_lr1e-3':'B3：随机初始 / 1e-3','C1_fixed_monotonic_lr1e-3':'C：固定单调时钟（补充）'}
    for arm in result['arms']:
        name=arm['arm']['name'];histories[name]=[json.loads(s) for s in (OUT/name/'history.jsonl').read_text().splitlines()]
    a=histories['A_small_overfit'];axes[0,0].plot([r['step'] for r in a],[r['metrics']['fit']['relative_rmse']*100 for r in a],color=colors[0],marker='o')
    axes[0,0].set(title='小样本拟合：128 个点 × 8 个时刻',ylabel='相对 RMSE（%）',xlabel='优化步数',yscale='log')
    for j,name in enumerate(list(histories)[1:]):
        h=histories[name]
        for axis,key in [(axes[0,1],'known_train'),(axes[1,0],'known_validation')]:
            axis.plot([r['step'] for r in h],[r['metrics'][key]['relative_rmse']*100 for r in h],label=labels[name],color=colors[j])
    axes[0,1].set(title='广域拟合：2048 个点 × 120 个时刻',ylabel='训练相对 RMSE（%）',xlabel='优化步数')
    axes[1,0].set(title='时间留出：相同点 × 30 个未拟合时刻',ylabel='时间留出相对 RMSE（%）',xlabel='优化步数')
    for j,(name,values) in enumerate(latent_data.items()):
        if name=='A_small_overfit':continue
        axes[1,1].plot(ts.numpy().ravel(),values['values'],label='原模型' if name=='source' else labels[name],color='#53616d' if name=='source' else colors[list(histories)[1:].index(name)],linestyle='--' if name=='source' else '-')
    axes[1,1].set(title='速度网络内部的一维时间变量',ylabel='内部时间标量 q(t)',xlabel='共同归一化时间')
    for axis in axes.flat:
        axis.grid(alpha=.2);axis.spines[['top','right']].set_visible(False)
    for axis in [axes[0,1],axes[1,0],axes[1,1]]:axis.legend(fontsize=8)
    fig.suptitle('冻结 H3 轨迹：速度拟合与内部时钟对照',fontsize=15,fontweight='bold')
    fig.savefig(REPORT/'velocity_fit_diagnostics.png',dpi=190);fig.savefig(REPORT/'velocity_fit_diagnostics.svg');plt.close(fig)

    fmt=lambda x: '-' if x is None else f'{x:.2f}'
    lines=['# 冻结 H3 轨迹：速度分支拟合实验','',
      '实验日期：2026-11-01。源模型为 H3 最终50k，单独训练诊断副本，不修改原模型或渲染器。','',
      '## 1. 实验干预与独立性','',
      '- 冻结 canonical 坐标、形变网络、hash grid；缓存完整渲染中心 R(q)X+b 的时间导数。h=0.001，端点用二阶单边差分。',
      '- 唯一目标为速度预测与缓存导数的归一化 MSE；没有 CMR、外力、本构、材料、RGB 或应力目标。',
      '- 使用原 PIDG VelocityNet 结构；训练共享特征、时间编码、ID embedding、速度头，冻结未使用的应力头参数。共享特征改变可能改变未使用的应力输出，所以该副本不能当作原应力模型直接部署。',
      f"- 训练点2048，独立留出点512；实际覆盖{manifest['unique_train_particle_ids']}个训练ID、{manifest['unique_heldout_particle_ids']}个留出ID，ID交集为0。先按ID家族分组，再采样opacity>=0.05的高斯，不是语义前景分割。",
      '- 120个训练时间与30个验证时间不交叉。后者是本轮速度拟合未见的时刻，仍由同一H3重建模型产生目标，不是物理真值。没有读取final test图像或元数据。',
      '- A为128点×8时刻、4000步全批量；B为2048点×120时刻、12000步、batch4096。所有模型按固定最终步数报告，不按验证择优挑选早期断点。',
      '- C为发现时间碰撞之后追加的诊断对照：使用B1相同目标/初始非时钟权重/随机采样序列/优化预算，将内部时钟固定为tanh(2t-1)。没有增加空间平面、ID维度或MLP容量。它是事后对照，时钟取值范围与条件性也随之改变，不能把全部差值唯一归为折返。',
      '- B3改变了初始化和随机种子，属于初始化敏感性检查，不能单独归因为初始化。C与B3部分并行运行，墙钟时间不是公平性能基准。',
      '- 初次按高斯点划分时发现增点共享ID，存在97个ID重叠；已在A的3000步停止并完整归档。下表仅使用修正后ID整组留出的重新实验。','',
      '## 2. 固定预算结果','',
      '相对误差为整体RMS(v预测-轨迹导数)/RMS(轨迹导数)，不是逐粒子百分比平均。零速度参考为100%。运动点按训练导数固定阈值筛选；角度不包含近零预测速度。','',
      '| 实验 | 拟合集相对误差 %↓ | 时间留出 %↓ | ID留出/训练时刻 %↓ | ID+时间留出 %↓ | 时间留出夹角中位数 °↓ | 时间留出速度比中位数→1 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    base=result['arms'][0]['initial']
    lines.append('| 原H3 | '+ ' | '.join([fmt(base['known_train']['relative_rmse']*100),fmt(base['known_validation']['relative_rmse']*100),fmt(base['heldout_train']['relative_rmse']*100),fmt(base['heldout_validation']['relative_rmse']*100),fmt(base['known_validation']['angle_median_deg']),fmt(base['known_validation']['speed_ratio_median'])])+' |')
    for arm in result['arms']:
        m=arm['final']['metrics'];name=arm['arm']['name']
        lines.append('| '+labels[name]+' | '+' | '.join([fmt(m['fit']['relative_rmse']*100),fmt(m['known_validation']['relative_rmse']*100),fmt(m['heldout_train']['relative_rmse']*100),fmt(m['heldout_validation']['relative_rmse']*100),fmt(m['known_validation']['angle_median_deg']),fmt(m['known_validation']['speed_ratio_median'])])+' |')
    if len(result['arms'])==5:
        b1=result['arms'][1]['final']['metrics'];c=result['arms'][-1]['final']['metrics']
        lines += ['', '### 直接发现','',
          f"小样本拟合误差降至{result['arms'][0]['final']['metrics']['fit']['relative_rmse']*100:.2f}%，排除了明显断梯度或完全不可训练。广域采样目标上，B1误差为{b1['known_train']['relative_rmse']*100:.2f}%（训练）/{b1['known_validation']['relative_rmse']*100:.2f}%（时间留出）；固定单调时钟C为{c['known_train']['relative_rmse']*100:.2f}%/{c['known_validation']['relative_rmse']*100:.2f}%。",
          '这些结果支持当前速度分支内部时间表示存在重要的混叠/条件性问题；这次没有修改数据集帧编号、共同时间轴或渲染输入时间。',
          f"C在ID+时间整组留出上仍为{c['heldout_validation']['relative_rmse']*100:.2f}%，空间/ID迁移尚未解决。训练家族与留出家族的空间分布和运动也可能不同，不能仅凭此把退步唯一归咎于embedding。",
          '外力在本实验中没有加入，也没有参与任何目标，因此这次改善不能归因为补上外力。时间输入修正仍需后续受控接入联合训练，才能评价图像质量和物理残差。']
    lines += ['','A的时间/ID留出包含大量从未参与小样本拟合的训练点，不与B的广域训练含义混同；A用于检查小样本拟合能力。',
      '', '![速度拟合诊断](velocity_fit_diagnostics.png)','',
      '## 3. 工程检查与目标可靠性','',
      '- 原诊断17项单元测试、时钟对照3项单元测试通过；两者各自100步真实CUDA smoke通过；未增加任何原训练路径钩子。',
      '- 形变/grid参数哈希不变、梯度为空；同一导数重复查询逐位一致。所有源权重/冻结代码/配置哈希在拟合前后相同，缓存目标SHA256不变；留出ID embedding逐位不变。',
      '- 速度分支首步梯度记录在各arm的result.json，所有受训参数组梯度有限。未使用的应力头参数逐位不变。',
      '- 原H3五个诊断时刻的relative RMSE得到复现，单时刻差异均小于0.0002。']
    for h in [.00025,.0005,.002]:
        vals=[r['relative_rmse']*100 for r in manifest['derivative_sensitivity'] if r['h']==h]
        lines.append(f'- h={h}相对h=0.001的导数变化：五个探针整体相对RMS {min(vals):.2f}% - {max(vals):.2f}%。这是数值步长敏感性，不是真实速度误差。')
    collision_path=REPORT/'time_collision_checks.json'
    if collision_path.exists():
        checks=read(collision_path)['results']
        lines += ['', '### 时间碰撞机制检查（事后）','']
        if 'B1_pretrained_lr1e-3' in checks:
            q=checks['B1_pretrained_lr1e-3']
            lines += [f"B1在t={q['t1']:.8f}与t={q['t2']:.8f}，内部时间值相差{q['encoded_time_absolute_difference']:.3g}，256个固定点的预测速度差RMS仅{q['predicted_vector_difference_rms']:.6g}，而冻结轨迹导数差RMS为{q['target_vector_difference_rms']:.6g}。这是几乎相同时间编码对应不同目标的直接例子。",
              f"若强制这两个时刻输出严格相同，仅这对查询的最佳相对RMSE下界为{q['exact_shared_output_pair_relative_rmse_lower_bound']*100:.2f}%。它不是完整训练集误差的下界，也不是物理速度真值检验。"]
        lines += ['单调时钟C不应出现这种折返碰撞。完整数值见[时间碰撞检查](time_collision_checks.json)。']
    lines += ['', '## 4. 可解释范围','',
      '- 小样本拟合成功只能排除严重断梯度/不可训练，不能证明网络可表示完整时间轨迹。',
      '- 广域训练误差与时间留出误差必须一起解释；低训练误差而留出差，倾向于时空插值/过拟合或ID依赖问题；训练也高则还需区分优化、时间表示和容量，不能单次宣布容量不足。',
      '- 时间编码先经Fourier，再投影压成一个标量，最终网络没有单独直接输入t。同一X和ID下，若该标量相同，速度预测相同。图中折返是时间混叠风险，仍需对照验证其具体贡献。',
      '- 重建和图像渲染完全冻结，因此本实验不产生新的PSNR提升，也不能单独证明缺少外力导致原始误差。轨迹导数是模型内部teacher，不是物理速度GT。','',
      '## 5. 复现与原始产物','',
      '```powershell',
      '. ./reproduction/scripts/enable_cuda.ps1 -ProjectRoot F:/A_Edinburgh/PhysGauss',
      'envs/pidg-cu128/python.exe -m variants.frozen_velocity_probe.run --phase export',
      'envs/pidg-cu128/python.exe -m variants.frozen_velocity_probe.run --phase smoke',
      'envs/pidg-cu128/python.exe -m variants.frozen_velocity_probe.run --phase train',
      'envs/pidg-cu128/python.exe -m variants.frozen_velocity_probe.time_control',
      'envs/pidg-cu128/python.exe -m variants.frozen_velocity_probe.collisions',
      'envs/pidg-cu128/python.exe -m variants.frozen_velocity_probe.report',
      '```','',
      '默认输出拒绝覆盖已有拟合历史；重跑请复制配置并指定新的output_run。','',
      '- [预定协议](protocol.md)、[配置](config.json)、[结果JSON](results.json)、[时间细节](temporal_details.json)',
      '- [追加时钟对照协议](clock_control_protocol.md)、[时钟对照结果](clock_control_result.json)、[时间碰撞检查](time_collision_checks.json)',
      '- [单测日志](unit_tests.log)、[CUDA短测](smoke.log)、[拟合日志](train.log)',
      '- 原始目标、哈希manifest和每组独立速度断点位于 `runs/pidg_frozen_velocity_probe/H3_final_20261101`。']
    (REPORT/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('REPORT COMPLETE',REPORT/'assessment.md')


if __name__=='__main__':main()
