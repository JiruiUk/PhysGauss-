"""Record reviewed negative coverage gate and preserve the complete 16k candidate."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.shape_coherence.run import RUNS,REPORTS,OUT,NAME,assert_frozen
from variants.metric_tuning import run as engine
from output.shape_coherence.continue_run import control_unchanged


def main():
    assert_frozen(ROOT/'variants/shape_coherence/code');control_unchanged()
    d=RUNS/NAME/'rubber_duck';p=d/'checkpoints/resume.pt';receipt=engine.read_json(p.with_suffix('.json'))
    assert receipt['iteration']==16000 and receipt['sha256']==engine.sha256(p)
    evidence=engine.read_json(REPORTS/'candidate_graph_evidence.json')
    assert evidence['checkpoint_sha256']==receipt['sha256']
    assert evidence['graph_sha256']==engine.sha256(d/'shape_reference/reference_graph.pt')
    side=torch.load(p.with_suffix('.shape.pt'),map_location='cpu',weights_only=True)
    assert side['checkpoint_sha256']==receipt['sha256'] and side['iteration']==16000
    assert engine.read_json(p.with_suffix('.metrics.json'))['shape_graph_sha256']==engine.sha256(p.with_suffix('.shape.pt'))
    from reproduction.scripts.training_checkpoint import load_state
    full=load_state(p)
    def finite(v):
        if torch.is_tensor(v):return bool(v.isfinite().all())
        if isinstance(v,dict):return all(finite(x) for x in v.values())
        if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
        return True
    assert all(finite(full[k]) for k in ['gaussian_parameters','velocity_net','grid4d','deform','gaussian_optimizer','deform_optimizer'])
    events=[json.loads(line) for line in (d/'shape_loss.jsonl').read_text().splitlines()]
    assert len(events)==1 and events[0]['event']=='graph_built' and events[0]['iteration']==16000
    graph=side['graph'];labels=graph['labels'];anchors=graph['anchors'];neighbors=graph['neighbors']
    assert (labels[anchors]>=0).all() and (labels[neighbors]==labels[anchors,None]).all()
    rows=evidence['per_frame'];mid=[r for r in rows if 62<=r['frame']<=108]
    decision={**evidence,'approved':False,'manual_review_pending':False,'decision_utc':engine.now(),
        'decision':'rejected_insufficient_visible_coverage_for_contact_target',
        'reason':'Early reference patches cover 64.6-67.3% of early duck contribution, but 7.29% at74 and zero at86/99/108. Visual review shows sparse colored fragments outside/around current visible bodies. No demonstrated coverage for the intended middle-stage shape repair.',
        'not_a_predefined_numeric_threshold':'The prospective gate was qualitative reliable coverage; no post-hoc numeric success threshold is claimed.',
        'manual_reviewed_training_frames':evidence['actual_train_only_frames'],
        'early_5_frame_components_reviewed':True,'no_shape_objective_evaluations_before_gate':True,
        'remaining_training_not_executed':34000,'total_additional_candidate_steps_executed':13000,
        'no_50k_test_or_benefit_claim':True,
        'causal_limit':'Visibility loss alone cannot distinguish occlusion/viewpoint changes, changing effective Gaussian footprint or correspondence drift. No true 3D material IDs, shape or contact forces are available.',
        'next_unexecuted_hypothesis':'Validate persistent object/surface support and full center-plus-footprint behavior over the training sequence before applying objectwise shape/contact constraints.',
        'no_new_candidate_launched':True}
    engine.write_json(REPORTS/'candidate_graph_review.json',decision)
    proof={'status':'passed','outcome':'graph_gate_rejected','manual_review_pending':False,
        'source_data_freeze_sha256':engine.sha256(REPORTS/'source_freeze.json'),
        'historical_control_SHA_unchanged':True,'common_start_sha256':engine.read_json(d/'branch.json')['identical_full_checkpoint_sha256'],
        'full_16k_checkpoint_sha256':receipt['sha256'],'graph_sha256':evidence['graph_sha256'],
        'checkpoint_sidecar_sha256':engine.sha256(p.with_suffix('.shape.pt')),
        'full_parameters_optimizer_finite':True,'graph_persisted_and_no_cross_inferred_label_edges':True,
        'shape_loss_evaluations':0,'candidate_steps_added':13000,'smoke_loss_evaluations':100,
        'smoke_optimizer_updates':99,'actual_flags_off_steps':40,
        'final_test_opened_for_this_candidate':False,'new_full_comparison_available':False}
    engine.write_json(REPORTS/'gate_verification.json',proof)
    canvas=Image.new('RGB',(1600,1340),'#182a3d');draw=ImageDraw.Draw(canvas)
    for row,number in enumerate([2,74,86,108]):
        with Image.open(OUT/'candidate_graph'/f'frame{number:04d}.png') as image:
            preview=image.convert('RGB').resize((1600,327))
        canvas.paste(preview,(0,row*335));draw.text((8,row*335+327),f'TRAIN frame {number}; left GT, middle candidate16k, right early reference labels',fill='white')
    canvas.save(OUT/'gate_review.png')
    blocks=''.join(f'<h2>训练帧 {r["frame"]}：参考点贡献 {100*r["known_contribution_on_duck_pixels"]:.2f}%</h2><img src="candidate_graph/frame{r["frame"]:04d}.png">' for r in rows)
    page='<!doctype html><meta charset="utf-8"><title>P1 形态监督覆盖检查</title><style>body{background:#17283b;color:white;font:18px system-ui;max-width:1600px;margin:24px auto}img{width:100%}p{line-height:1.7}</style><h1>P1：16k覆盖门槛未通过</h1><p>尚未启用新形态损失。每行左：训练GT；中：16k候选；右：早期参考点颜色，未知点为实际颜色的灰暗版本。不是训练完成后的效果比较，不是真实物体/材料分割。</p><p>早期贡献约65–67%，帧74降至7.29%，帧86/99/108为零。未能证明这组监督覆盖待修复的中段可见形态，因此保留16k完整状态，停止余下34k，不宣布改进成功。可见贡献不足不能单独证明材料身份错误，亦不能归因缺失外力。</p><p><a href="../c1_flow_ablation/C1_repaired/index.html">历史C1完整GT|Render（未重训）</a></p>'+blocks
    (OUT/'index.html').write_text(page,encoding='utf8')
    state=engine.read_json(RUNS/'controller.json');state.update(status='rejected_graph_gate',updated_utc=engine.now(),
        stopped_iteration=16000,decision_file=str(REPORTS/'candidate_graph_review.json'))
    engine.write_json(RUNS/'controller.json',state)
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
