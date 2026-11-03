"""Reuse the verified native sequence exporter with accurate R0 labels."""
import importlib.util
from .run import ROOT, REPORT, NAMES, engine


def main():
    spec=importlib.util.spec_from_file_location('r0_existing_export',ROOT/'experiments/transfer_integrity/export.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    e.REPORT=REPORT;e.OUT=ROOT/'output/static_pool';e.NAMES=['S0_previous_control',*NAMES]
    e.main()
    p=e.OUT/'index.html';text=p.read_text(encoding='utf-8')
    text=text.replace('Transfer integrity 第一版测试','R0 静态独立表示测试')
    text=text.replace('第一版：真实观测、短时对应与局部几何先验','R0：可靠静态子集的时间不变表示')
    text=text.replace('左：原始 C1 50k；中：继续训练的 U0 52k；右：新约束 U1 52k。',
        '左：上一轮S0（仅参考）；中：本轮匹配H0对照；右：本轮H1静态池。全部为最终50k状态。')
    text=text.replace('../../experiments/transfer_integrity/assessment.md','../../experiments/static_pool/assessment.md')
    p.write_text(text,encoding='utf-8')
    engine.write_json(REPORT/'artifact_manifest.json',{str(q.relative_to(ROOT)):engine.sha256(q) for q in e.OUT.rglob('*')
                                                      if q.is_file() and '__pycache__' not in str(q)})


if __name__=='__main__':main()
