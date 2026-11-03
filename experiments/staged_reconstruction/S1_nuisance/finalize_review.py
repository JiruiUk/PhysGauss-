"""Persist the completed negative manual review without editing frozen gates."""
from datetime import datetime,timezone
import difflib
import json
from pathlib import Path
import subprocess
from variants.staged_reconstruction.nuisance_static.prepare import ROOT,REPORT,RUNS,OUT,verify
from variants.staged_reconstruction.prepare_data import read,write,sha


def main():
    verify(); gate = read(REPORT/'synthetic_gate.json'); exports = read(REPORT/'synthetic_exports.json')
    assert not gate['numerical_passed'] and exports['passed']
    assert not (REPORT/'final_review.json').exists()
    evaluation = read(REPORT/'synthetic_evaluation.json')
    for path,expected in evaluation['checkpoint_sha256'].items(): assert sha(ROOT/path) == expected,path
    for record in list(exports['PLY'].values())+list(exports['videos'].values())+list(exports['depth_alpha'].values()):
        assert sha(ROOT/record['path']) == record['sha256']
    pictures = list(OUT.glob('*.png'))
    receipt = {'completed_at':datetime.now(timezone.utc).isoformat(),'status':'complete_rejected',
        'manual_review_completed':True,'promoted':False,'real_training_started':False,
        'rejection':'Both fixed synthetic cases fail static-GT PSNR and RGB spill gates; unchanged thresholds.',
        'actually_viewed':['diffuse/glossy static native c0/3/8/15/24/31',
            'diffuse/glossy train decomposition0/24/48/72','diffuse/glossy D/alpha and alpha c3/24/31',
            'diffuse/glossy decoded160-frame video contacts'],
        'findings':'Column/thin plate retained; detached blocks/base halos/alpha floaters remain; nuisance also explains some static appearance.',
        'no_real86_188_export_reason':'Synthetic gate failed before real continuation; no new real model exists.',
        'cautions':['0.05 coverage tolerance exceeds0.04 plate thickness',
            'D/alpha and normal errors use conditional, arm-dependent valid pixel sets',
            'Dark RGB does not establish empty geometry',
            'Per-image nuisance is not physical identity; preserve S3-U and unknowns'],
        'checks':{'tests':72,'four_full_state_audits':True,'matched6000_camera_sequences':True,
            'PLY_exact_finite':True,'two_videos_decoded_frames_each':160,'browser_playback_tested':False},
        'evidence_sha256':{p.name:sha(p) for p in [REPORT/'synthetic_evaluation.json',REPORT/'synthetic_gate.json',
            REPORT/'synthetic_exports.json',REPORT/'source_freeze.json']},
        'reviewed_image_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in pictures},
        'next':'Read-only audit of existing RGB-derived depth reliability before a separate bounded surface-constraint experiment; do not repeat S1-N.'}
    write(REPORT/'final_review.json',receipt)
    write(RUNS/'final_status.json',{'status':'complete_rejected','real_training_started':False,
        'review':(REPORT/'final_review.json').relative_to(ROOT).as_posix(),'review_sha256':sha(REPORT/'final_review.json')})
    section = '''

### S1-N：完整Gaussian合成验证未通过（2026-11-03）

完成漫反射/反光×全RGB对照/每图干扰层四臂，各6000更新；72测试、原生
flags-off/梯度、双进程恢复、完整Adam/RNG/同6000相机序列及冻结审计通过。
未见相机独立静态GT PSNR对照→干扰34.4082→33.6690、33.8767→33.2329dB，
RGB外溢增加6.05%/6.44%。可见点表面覆盖略升，薄板约99%，但0.05容差
大于0.04厚度，不能证明正确厚度。条件D/alpha略好不抵消静态图像退化。
实际静态/分解/深度-alpha图仍见方块残留、底座光晕和悬浮片；两视频各完整
解码160帧，四PLY精确对应检查点。合成相机时间网格不是物理运动轨迹。

拒绝本版升级，未启动真实18k→24k，不替换S3-U，不改冻结权重/门槛。
下一步先只读审计已有RGB推断深度/遮挡/时间/薄板证据，避免把错误深度作为
强表面约束；再选择一个独立开关的表面约束候选。暂不能断言数据极限。
不重复C1/S1-N，不扫权重/门槛，不恢复旧暂停任务，不引入颜色/静态标签、
数量、材质、本构、接触或力。见[S1-N实测评估](staged_reconstruction/S1_nuisance/v1/assessment.md)。
'''
    patches = []
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:
        original = subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
        path = ROOT/name; working = path.read_text(encoding='utf-8').replace('\r\n','\n')
        assert '### S1-N：完整Gaussian合成验证未通过' not in working
        path.write_text(working+section,encoding='utf-8',newline='\n')
        patches.extend(difflib.unified_diff(original.splitlines(True),(original+section).splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (ROOT/'tmp/S1_nuisance_plans_cached.patch').write_text(''.join(patches),encoding='utf-8',newline='\n')
    verify(); print('S1_N_REVIEW_COMPLETE_REJECTED')


if __name__ == '__main__': main()
