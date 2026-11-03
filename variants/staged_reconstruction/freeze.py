"""Keep formal S1 source and inputs fixed; no changes to historical code."""
from .prepare_data import ROOT, REPORT, RUNS, read, write, sha


def verify():
    expected=read(REPORT/'S1_source_freeze.json')
    for path,digest in expected.items():
        assert sha(ROOT/path)==digest, 'Frozen S1 source/input changed: '+path


def main():
    files=list((ROOT/'variants/staged_reconstruction').glob('*.py'))
    files += [ROOT/'tests/test_staged_static.py',REPORT/'S1_protocol.json',REPORT/'camera_audit.json',
              REPORT/'data_receipt.json',RUNS/read(REPORT/'S1_protocol.json')['initialization_file']]
    # Include every inherited baseline source/input already protected by R0.
    previous=read(ROOT/'experiments/static_pool/training_freeze.json')
    for group,items in previous.items():
        base=ROOT/{'fork':'variants/static_pool/code','C1_parent':'variants/color_repair/code'}.get(group,'')
        for name,digest in items.items():
            assert sha(base/name)==digest
            files.append(base/name)
    values={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in files}
    destination=REPORT/'S1_source_freeze.json'
    if destination.exists():assert read(destination)==values
    else:write(destination,values)
    verify();print('S1 source/input freeze verified:',len(values),flush=True)


if __name__=='__main__':main()
