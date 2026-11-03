"""Independent S2 source/input lock; never changes the frozen S1 file set."""
from pathlib import Path
from ..prepare_data import ROOT, REPORT, RUNS, read, write, sha


def verify():
    value=read(REPORT/'S2_source_freeze.json')
    for name,expected in value['sha256'].items():assert sha(ROOT/name)==expected,name
    from ..freeze import verify as verify_static
    verify_static()


def main():
    files=list(Path(__file__).parent.glob('*.py'))+[REPORT/'S2_protocol.json',REPORT/'S2_identity.json',REPORT/'S2_data_receipt.json',RUNS/'S2_observations.npz']
    files += [RUNS/o['file'] for o in read(REPORT/'S2_data_receipt.json')['objects']]
    files += [ROOT/p for p in ['tests/test_staged_objects.py','output/surface_support/core.py','variants/flow_comparison/geometry.py']]
    value={'sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in files}}
    path=REPORT/'S2_source_freeze.json'
    if path.exists():assert value==read(path),'Existing S2 freeze differs; preserve attempts and version the new protocol'
    else:write(path,value)
    verify()


if __name__=='__main__':main()
