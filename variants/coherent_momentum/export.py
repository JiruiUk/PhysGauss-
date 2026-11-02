"""Verified pair exporter and prospective M1 comparison."""
import json
from variants.flow_comparison import export as old
from variants.metric_tuning.run import read_json as read,write_json as write,sha256
from .run import ROOT,REPORTS,RUNS,NAME
from .evaluate import evaluate,CONTROL
OUT=ROOT/'output/coherent_momentum'


def export(directory):
    # Precreate serially to avoid the Windows concurrent mkdir failure seen in M1.
    for tag in ['best','50000']:(OUT/NAME/'pairs'/tag).mkdir(parents=True,exist_ok=True)
    old.REPORTS,old.RUNS,old.OUT=REPORTS,RUNS,OUT
    old.export(directory)


def report():
    rows,physics=evaluate();run=RUNS/NAME/'rubber_duck'
    a=read(CONTROL/'metric_best.json')['PSNR'];b=read(RUNS/'selection_lock.json')['models'][NAME]
    prior=read(ROOT/'experiments/momentum_repair/motion_validation.json')['results'][0]['metrics']
    mm=rows[0]['metrics']
    delta={'validation_PSNR':b['value']-a['value'],
        'validation_LPIPS_relative':b['metrics']['LPIPS']/a['metrics']['LPIPS']-1,
        'geometric_EPE_relative':mm['geometric_EPE_px']/prior['geometric_EPE_px']-1,
        'velocity_EPE_relative':mm['velocity_EPE_px']/prior['velocity_EPE_px']-1}
    p=read(REPORTS/'plan.json')['success_prospective']
    criteria={'image_PSNR':delta['validation_PSNR']>=-p['validation_PSNR_drop_at_most_dB'],
        'image_LPIPS':delta['validation_LPIPS_relative']<=p['validation_LPIPS_relative_worse_at_most'],
        'observed_motion':delta['velocity_EPE_relative']<=p['independent_velocity_EPE_relative_worse_at_most']}
    for row in physics['paired_comparisons']:
        criteria[row['operator']+'_momentum']=row['relative_change']<=-.25
        criteria[row['operator']+'_coverage']=row['min_paired_valid_fraction']>=.5
    decision={'criteria':criteria,'composite_benefit':all(criteria.values()),'deltas_N1_minus_M1':delta,
        'no_further_automatic_full_training':True,'not_physical_ground_truth':True}
    write(REPORTS/'decision.json',decision)
    write(REPORTS/'results.json',{'metrics':read(run/'results.json'),'control_metrics':read(CONTROL/'results.json'),
        'selection':b,'control_selection':a,'decision':decision,'motion':rows,
        'physical_operator_comparison':physics,'protocol':read(REPORTS/'plan.json')})
    lines=['# 统一离散轨迹动量：N1 与 M1','',
        'L1/M1已经共享完整中心轨迹；N1进一步统一时间区间并替换空间求导算子，未新增独立速度网络。',
        '原stress_pred、率修正系数和M1梯度策略保留；没有新参数、外力或本构。共同3k至50k，单场景单种子，历史测试已暴露。',
        '同时改变时间差分精度、局部空间阶次、邻居数和求解精度，不声称仅改变一个标量，也不将结构恒等视作真值。','',
        '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','',
        '最终解释须区分图像质量、独立二维观测和两种离散算子的动量残差；同算子同材料点比较。',
        '物理推进位置误差与加速度残差只差h²，因此不是独立证据。边界区间中心偏移和无效邻域须披露。',
        '自动报告已生成，交付前还需独立SHA、像素、视频、ZIP和人工审阅。','']
    (REPORTS/'assessment.md').write_text('\n'.join(lines),encoding='utf8')
