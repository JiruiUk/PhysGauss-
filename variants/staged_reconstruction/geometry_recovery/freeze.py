from pathlib import Path
from ..prepare_data import ROOT,read,write,sha
from .model import GEOM_REPORT,GEOM_RUNS


def verify():
    for name,digest in read(GEOM_REPORT/'source_freeze.json').items():assert sha(ROOT/name)==digest,name
    from ..recovery.freeze import verify as previous
    previous()


def main():
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_staged_geometry_recovery.py',GEOM_REPORT/'S2_protocol.json',GEOM_REPORT/'S2_data_receipt.json']
    files += [GEOM_RUNS/o['file'] for o in read(GEOM_REPORT/'S2_data_receipt.json')['objects']]
    values={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files};file=GEOM_REPORT/'source_freeze.json'
    if file.exists():assert read(file)==values
    else:write(file,values)
    verify()


if __name__=='__main__':main()
