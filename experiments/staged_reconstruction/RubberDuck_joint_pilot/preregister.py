"""Freeze the one real pilot before implementation or formal updates."""
import difflib
import subprocess
from variants.staged_reconstruction.prepare_data import ROOT, read, sha, write

BASE = ROOT/'experiments/staged_reconstruction/RubberDuck_joint_pilot'

def main():
    assert not (BASE/'protocol.json').exists()
    p = {'stage': 'J4 RubberDuck joint reconstruction pilot', 'default_enabled': False,
        'human_requested_real_pilot': True, 'J3_automatic_geometry_promotion': False,
        'parent': 'runs/pidg_staged_reconstruction/S3_union/physics/formal/checkpoint.pt',
        'parent_sha256': '8c767b7835eee936e49989369d7bc0277d7567d3aedaa7385c06654284b68ad0',
        'static_parent': 'runs/pidg_staged_reconstruction/S1_static/checkpoint.pt',
        'static_parent_sha256': sha(ROOT/'runs/pidg_staged_reconstruction/S1_static/checkpoint.pt'),
        'arms': ['frozen_static', 'joint_depth_trust'], 'updates': 2000, 'width': 1600,
        'sampling': 'continue inherited Python RNG and remaining120-view stack; refill random.sample(range(120),120)',
        'RGB_loss': '0.8 L1 + 0.2 DSSIM full image; RGBA composited black only',
        'self_depth_weight': .1, 'self_depth': 'same frozen J3 loss; opacity-only static render; parent alpha>.5',
        'static_geometry_frozen': True, 'static_appearance': ['_opacity','_features_dc','_features_rest'],
        'physics': 'frozen; original complete Adam and graph cache preserved; no physics losses',
        'Adam_eps': 1e-15, 'dynamic_rates': 'exact original union.optimizer and parent full state',
        'static_rates': 'exact original optimizer_for and S1 full state; no reset',
        'topology_changes': False, 'legacy_conditional_color_masks': True,
        'new_semantic_masks_GT_depth_normals_object_counts': False,
        'qualification_steps': 20, 'recovery_native_gradient_atol': 1e-6,
        'recovery_native_gradient_rtol': 1e-5, 'recovery_state_and_fixed_gradient_Adam': 'exact',
        'checkpoint_every': 250, 'fixed_cameras': [86,188], 'fixed_times': [2,100,200],
        'video_times': list(range(1,201)), 'train_indices': [0,24,48,72,96,119],
        'validation_split': 'all30; native1600; no semantic masks; test withheld',
        'validation_PSNR_max_drop_dB': .3, 'manual_alpha_depth_cleanup_and_completeness_required': True,
        'fixed_view_independent_GT': False, 'teacher_is_independent_GT': False,
        'original_synthetic_gates_unchanged': True, 'seed_reset': False}
    assert sha(ROOT/p['parent']) == p['parent_sha256']
    write(BASE/'protocol.json', p)
    write(BASE/'preregistration.json', {'files': {x.relative_to(ROOT).as_posix(): sha(x)
        for x in [BASE/'plan.md',BASE/'protocol.json']}})
    section = '''### J3完成与Rubber Duck联合试验J4（2026-11-04）

J3已5a6a61a8完成两新候选各6000步，匹配历史控制未重复。原数值门槛全部通过：
静态RGB外溢下降41.13/40.10%，准确射线覆盖净+49/+27；但原生alpha/D人工
复核仍有悬浮与底座晕。独立事后支撑审计outside-GT高alpha仅减2.39/2.59%，
不能将变暗当几何清理，最终completed_not_promoted/pending:false。8PLY与
两H264各160完整解码已核验；这是合成Gaussian，不是Rubber Duck结果。

用户明确要求回Rubber Duck检测并联合动态物体提升。另立J4有界真实探索，
experiments/staged_reconstruction/RubberDuck_joint_pilot/plan.md及protocol.json。
同一S3-U final6k完整父与其自身静态库：冻结静态对照/开放opacity-SH并加
固定0.1父D/alpha自身参照候选，各2000步；不嫁接S1-C、不改J3失败结论。
完整继承动态Adam/RNG/相机与S1静态Adam（静态权重已逐值相同），静态几何
冻结。physics参数/Adam/图保留且冻结，纯RGB重建试验不能称原物理续训。
父曾用颜色条件掩膜须披露，新训练无新语义掩膜/GT/数量。先独立完整恢复
资格再训练，原生86/188早中晚及全序列RGB/alpha/D和全部val比较，缺少独立
固定视角GT须明说，几何/完整性人工审计决定结果，不按PSNR自动升级。'''
    patches=[]
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:
        head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        path=ROOT/name; current=path.read_text(encoding='utf-8')
        assert section.splitlines()[0] not in current
        path.write_text(current.rstrip()+'\n\n'+section+'\n',encoding='utf-8',newline='\n')
        updated=head.rstrip()+'\n\n'+section+'\n'
        patches.extend(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    dest=ROOT/'runs/pidg_staged_reconstruction/RubberDuck_joint_pilot/v1/plans_cached.patch'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(''.join(patches),encoding='utf-8',newline='\n')
    print('J4_PREREGISTERED',sha(BASE/'protocol.json'),dest)

if __name__=='__main__': main()
