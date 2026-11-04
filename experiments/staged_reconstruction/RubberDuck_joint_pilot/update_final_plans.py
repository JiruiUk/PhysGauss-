"""Append only final J4 evidence using cached-only main-plan patches."""
import difflib
import subprocess
from variants.staged_reconstruction.rubber_joint import runtime as r
from variants.staged_reconstruction.prepare_data import ROOT,read

SECTION='''### J4 Rubber Duck联合对照完成（2026-11-04）

9f5d54dd预声明，3f1d8974实施，88ccffb9资格通过，两臂各2000新更新完整完成。
原生1600的30张val平均PSNR父28.1663→冻结静态30.1266→联合31.3365dB，
联合较匹配续训多1.2099dB；SSIM0.974798→0.978989，属于RGB改善。
沿用S3-U自身4126静态/15507统一动态池，无数量/身份/新语义掩膜。历史父曾用
颜色条件掩膜必须保留披露；不是从零未知静态重建，也不是物理采集数据声明。
完整继承静态12k Adam（候选opacity/SH至14k）及动态22k→24k Adam、RNG/
同2000相机采样。静态几何、velocity/stress/其600步Adam及图缓存精确不变；
本轮纯RGB重建不能称原physics续训。10测试/198输入源码冻结/原生flags-off
梯度及两独立进程完整双Adam/RNG/相机/图与固定native梯度Adam更新精确通过。

实际复核86/188早中晚分解与原生1600RGB、alpha/D和六训练轨迹观测：静态边缘
和轨迹图像更清楚，但离开训练相机-时间组合的鸭子仍碎裂，动态暗RGB区域
仍有大块alpha/D拖尾，未通过干净支撑/完整性人工门槛。c86/c188父静态高alpha
支持135441/148933→134111/147546，父支持内D/alpha变化0.002774/0.002762，
只是自身代理，不是独立GT几何准确率。保留有限RGB收益，completed_not_promoted。
原生同相机全200时间视频/PLY与检查点核验见RubberDuck_joint_pilot/v1报告；
不重跑J4、不自动延长或升权，不改J3失败结论。Rubber Duck仍是下一步目标：
先审计实际相机-时间覆盖和动态暗色高alpha支撑的RGB可见性，复用已有耦合/
完整性研究，不能从暗色/残差直接删点或伪造同时间多视角GT。下一候选独立固定
协议后再实施，不回到无止境辅助深度循环。'''

def main():
    final=read(r.REPORT/'final_review.json');assert not final['pending_review']
    patches=[]
    for name in ['experiments/adaptive_reconstruction_plan.md','experiments/staged_reconstruction_plan.md']:
        head=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT).decode('utf-8')
        path=ROOT/name;current=path.read_text(encoding='utf-8');assert SECTION.splitlines()[0] not in current
        updated=head.rstrip()+'\n\n'+SECTION+'\n'
        path.write_text(current.rstrip()+'\n\n'+SECTION+'\n',encoding='utf-8',newline='\n')
        patches.extend(difflib.unified_diff(head.splitlines(True),updated.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    dest=r.RUNS/'final_plans_cached.patch';dest.write_text(''.join(patches),encoding='utf-8',newline='\n');print(dest)

if __name__=='__main__':main()
