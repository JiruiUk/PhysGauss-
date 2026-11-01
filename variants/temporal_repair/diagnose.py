"""Summarize worst-frame evidence; separate exposed test diagnostics from training inputs."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'experiments/temporal_repair'


def diagnose():
    source=json.loads((ROOT/'output/pdf/integrity_analysis_assets/data.json').read_text(encoding='utf-8'))
    models={m['id']:m for m in source['models']};h=models['H3'];b=models['C1']
    test=[]
    for file,psnr in sorted(h['per_view']['PSNR'].items(),key=lambda item:item[1])[:7]:
        i=int(Path(file).stem);test.append({'frame':i*4+1,'index':i,'PSNR':psnr,'delta_vs_C1':psnr-b['per_view']['PSNR'][file],
            'integrity':h['integrity']['per_frame'][i]})
    history=[json.loads(s) for s in (Path(h['directory'])/'integrity_history.jsonl').read_text().splitlines()]
    validation=sorted(history[-1]['per_frame'],key=lambda r:r['opaque_appearance_error'],reverse=True)[:7]
    readout=json.loads((REPORTS/'readout_validation.json').read_text(encoding='utf-8'))
    rows={k:{'metrics':v['metrics'],'worst5_PSNR':v['worst5_PSNR'],'frame84':next(r for r in v['per_frame'] if r['frame']=='0084')} for k,v in readout['models'].items()}
    result={'test_used_for_diagnosis_only':test,'validation_appearance_failures':validation,'validation_readout_ablation':rows,
        'confirmed_code_fact':'Renderer adds d_rotation to normalized canonical quaternion; CUDA computeCov3D does not normalize the sum.',
        'code_locations':['variants/integrity_tuning/code/gaussian_renderer/__init__.py:scales/rotations',
            'third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/cuda_rasterizer/forward.cu:127'],
        'causal_limit':'Nonunit covariance rotations are a concrete parameterization risk, not proof of the unique cause of frame85.',
        'rejected_change':'Simple posthoc center interpolation worsens validation frame84 and mean/tail metrics; no interpolation prior enabled.',
        'next_ablations':['unit covariance rotations with retraining','train-only foreground-error EMA frame sampling','both'],
        'training_input_policy':'Only train RGB/alpha used; validation chooses; original test already exposed; no baseline weights overwritten'}
    (REPORTS/'diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# H3 最差帧诊断与修正依据','',
        '测试集只用于已暴露问题诊断；新损失与采样策略只使用120个训练帧。验证集30帧用于方案与快照选择。','',
        '## 最差测试帧（按PSNR从低到高）','',
        '| 原始帧号 | H3 PSNR | 相对C1 | 内部低覆盖50 % | 不透明外观错误 % |','|---|---:|---:|---:|---:|']
    for r in test:lines.append(f"| {r['frame']} | {r['PSNR']:.4f} | {r['delta_vs_C1']:+.4f} | {100*r['integrity']['core_low_50']:.4f} | {100*r['integrity']['opaque_appearance_error']:.4f} |")
    lines+=['','## 判断','',
        '帧85的严格内部低覆盖约0.10%，而不透明外观错误约3.36%；主要可见问题包含错位、形变和颜色拖影，不能当作单纯透明度不足。验证帧84也有同类问题。',
        '相邻训练时刻的线性插值不能解决此问题：中心50%插值令验证均值PSNR下降约0.102dB，最差5帧均值下降约0.393dB，帧84从29.9108降至29.2976dB。完整插值的结果也单独记录，不作为已证实修正。',
        '明确实现风险：Gaussian旋转增量与canonical单位四元数相加后不再保证单位长度；CUDA协方差计算并不归一化它。新增独立config开关，在新训练中归一化；不能直接修改已有权重后声称提升。',
        '第二项修正是训练帧级困难采样：8k之后，25%的采样机会转向训练前景RGB误差EMA最高的25%训练帧，其余保留原始stack采样。完全不根据测试/验证帧误差决定训练采样。','',
        '## 受控验证','',
        'R1：H3+单位协方差旋转；R2：H3+困难训练帧采样；R3：两者组合。均复用同一个随机训练3k完整起点，总计50k，不从H3的50k模型续训。',
        'H3复用，不重新训练。最终是否改进依据验证均值、四项图像指标护栏、最差5帧PSNR及覆盖/背景指标判断；没有合格新模型时继续保留H3。',
        '局限：单场景单种子，测试集历史暴露，完整度是2D指标，尚不能确定唯一因果。']
    (REPORTS/'diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Export explicit GT/native-H3/smoothed comparison for the analogous validation failure.
    native=readout['models']['native']['per_frame'];i=next(x['index'] for x in native if x['frame']=='0084')
    raw=ROOT/'runs/pidg_temporal_repair/readout_ablation/validation';folder=REPORTS/'diagnosis_images';folder.mkdir(exist_ok=True)
    panel=[('GT',raw/'gt'/f'{i:05d}.png'),('H3',raw/'native/renders'/f'{i:05d}.png'),('center half (rejected)',raw/'center_half/renders'/f'{i:05d}.png')]
    image=Image.new('RGB',(4800,950),'#edf2f7');draw=ImageDraw.Draw(image);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',26)
    for j,(label,path) in enumerate(panel):
        with Image.open(path) as im:image.paste(im,(j*1600,50))
        draw.text((j*1600+12,10),label+' / validation frame0084',font=font,fill='#17304a')
    image.save(folder/'validation_0084_gt_h3_interpolation.png')


if __name__=='__main__':diagnose()
