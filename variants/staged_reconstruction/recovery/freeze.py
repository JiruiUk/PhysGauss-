from pathlib import Path
from ..prepare_data import ROOT,read,write,sha
from .prepare import REC_REPORT,REC_RUNS


def verify():
    for name,digest in read(REC_REPORT/'source_freeze.json').items():assert sha(ROOT/name)==digest,name
    from ..stage2.freeze import verify as previous
    previous()


def main():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_staged_recovery.py',REC_REPORT/'S2_protocol.json',REC_REPORT/'S2_data_receipt.json']
    paths += [REC_RUNS/o['file'] for o in read(REC_REPORT/'S2_data_receipt.json')['objects']]
    value={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths};path=REC_REPORT/'source_freeze.json'
    if path.exists():assert read(path)==value
    else:write(path,value)
    verify()


if __name__=='__main__':main()
