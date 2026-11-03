"""Reuse established shape/video checks and export accurate cold-start labels."""
from pathlib import Path
import importlib.util
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from variants.metric_tuning.runtime import read_json,write_json,sha256
spec=importlib.util.spec_from_file_location('v1_export',ROOT/'experiments/transfer_integrity/export.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
e.REPORT=ROOT/'experiments/scratch_integrity';e.OUT=ROOT/'output/scratch_integrity'
e.NAMES=['C1_original_50k','S0_scratch_control','S1_scratch_transfer']


def main():
    e.main();p=e.OUT/'index.html';text=p.read_text(encoding='utf-8')
    text=text.replace('Transfer integrity 第一版测试','Cold-start transfer integrity 第二版测试')
    text=text.replace('第一版：真实观测、短时对应与局部几何先验','第二版：从零训练、共同预热与几何约束')
    text=text.replace('左：原始 C1 50k；中：继续训练的 U0 52k；右：新约束 U1 52k。','左：历史C1 50k（参考）；中：新种子从零训练S0 50k；右：同起点新约束S1 50k。')
    text=text.replace('../../experiments/transfer_integrity/assessment.md','../../experiments/scratch_integrity/assessment.md')
    text='\n'.join(line.rstrip() for line in text.splitlines())+'\n';p.write_text(text,encoding='utf-8')
    write_json(e.REPORT/'artifact_manifest.json',{str(q.relative_to(ROOT)):sha256(q) for q in e.OUT.rglob('*') if q.is_file() and '__pycache__' not in str(q)})
    v=read_json(e.REPORT/'export_verification.json');v['adapter_sha256']=sha256(Path(__file__));write_json(e.REPORT/'export_verification.json',v)


if __name__=='__main__':main()
