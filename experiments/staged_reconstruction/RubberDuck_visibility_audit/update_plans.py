"""Task-only cached patches preserve concurrent plan and mapping edits."""
import difflib
import subprocess
from variants.staged_reconstruction.rubber_visibility import worker as w

SECTION='''### J5 Rubber Duck可见性审计完成（2026-11-04）

c434fc1c预声明、72e9dd22实现/资格通过。三个既有检查点各120源原生1600只读
审计完成，0训练更新；完整模型/双Adam/RNG/相机/physics图缓存前后逐值不变。
实际train120张=120精确时间=120相机变换，同时间多相机时间数0，各相机仅1时刻。
固定86/188早中晚18目标和源0/24/72原RGB/原生贡献、全120聚合分母/图像SHA
独立重放通过。6测试，原默认关/打包与单独渲染/原生RGB-alpha-D一致，两进程
资格逐值一致。单高斯确认导出weight是post-T的alpha*(1-alpha)，不用它量贡献。
新夹具主点错误保留日志和冻结代码，独立文件修复，未改协议/门槛或PIDG。

J4联合六组在目标暗高alpha区域的first20 ID并集解释99.83%—99.99%原生alpha，
但同组沿训练轨迹可见贡献约75.88%—88.48%落在暗观测，约11.52%—24.12%仍落
在较亮观测。组级非零不等于每点可见，暗观测不等于背景/可删，亮观测不保证
正确表面/物理身份。86/200暗高alpha像素父17,788、冻结39,830、联合37,689，
全像素分母1,440,000；各模型集合不同，不能叫GT伪影面积。实际看6张三模型
固定对照、9张训练源贡献图、2张原生1600 RGB/alpha，宽大暗支撑/碎裂仍在。
J5 completed_read_only_not_promoted/pending:false，没有新训练/视频/PLY或独立GT。
报告：RubberDuck_visibility_audit/v1/assessment.md、independent_audit.json和manual_review.json。

下一机制采用时间变可靠锚点约束弱可见动态运动，复用uncertainty_motion的2195
RGB支持锚点证据；它不是深度GT。并行“动态物体保持完整度”已在reliable_motion
实际实现该机制，本线不重复/不停止它，不自动拼接未通过候选或重启J4。
先跟随其完整资格、匹配训练和原6×9/6×200完整度结果，通过后另立静态
opacity/SH可撤销联合的完整状态匹配协议；原结构/外观门槛保持，不按暗色删点。
父历史颜色条件掩膜/统一动态池未知数量继续披露，无新物体数/静态标签/GT输入。
这不证明RGB无解，整体清理与完整度目标仍未完成。'''

MAP='''### J5 Rubber Duck read-only visibility audit

- `variants/staged_reconstruction/rubber_visibility/{core,worker}.py`: diagnostic
  subset/denominators, exact camera-time census and original native override-color
  visibility on three pinned states; no optimizer or baseline changes.
- `experiments/staged_reconstruction/RubberDuck_visibility_audit/`: frozen plan,
  inputs, preserved failed evaluation fixture, independent repair and metadata/
  fixed-source replay, actual image review and final assessment. Native first20
  IDs select a capacity-limited union; post-T exported weights are never treated
  as compositing contributions. Historical color-conditioned parent disclosed.
- `tests/test_rubber_visibility*.py`: six tests. Default off; full state unchanged.
  This maps a diagnostic only, not new Gaussian training or physics integration.'''

ATTR='''# Preserve J5 read-only audit/failure receipts and frozen input code bytes.
variants/staged_reconstruction/rubber_visibility/** -text
experiments/staged_reconstruction/RubberDuck_visibility_audit/** -text
tests/test_rubber_visibility.py -text
tests/test_rubber_visibility_repair.py -text'''


def main():
    assert not w.read(w.REPORT/'final_review.json')['pending_review'];patches=[]
    sections={'experiments/adaptive_reconstruction_plan.md':SECTION,
        'experiments/staged_reconstruction_plan.md':SECTION,'PIDG_REPO_MAP.md':MAP,'.gitattributes':ATTR}
    for name,section in sections.items():
        head=subprocess.check_output(['git','show','HEAD:'+name],cwd=w.ROOT).decode('utf-8')
        assert section.splitlines()[0] not in head
        path=w.ROOT/name;current=path.read_text(encoding='utf-8')
        if section.splitlines()[0] not in current:
            path.write_text(current.rstrip()+'\n\n'+section+'\n',encoding='utf-8',newline='\n')
        target=head.rstrip()+'\n\n'+section+'\n'
        patches.extend(difflib.unified_diff(head.splitlines(True),target.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    destination=w.RUNS/'final_plans_cached.patch';destination.write_text(''.join(patches),encoding='utf-8',newline='\n')
    print(destination)


if __name__=='__main__':main()
