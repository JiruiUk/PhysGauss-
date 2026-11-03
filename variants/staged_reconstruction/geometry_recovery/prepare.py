import argparse
import shutil
from ..prepare_data import ROOT,REPORT,read,write,sha
from ..recovery.prepare import REC_REPORT,REC_RUNS
from .model import GEOM_REPORT,GEOM_RUNS


def main():
    p=argparse.ArgumentParser();p.add_argument('--enable-geometry-recovery',action='store_true');args=p.parse_args()
    if not args.enable_geometry_recovery:raise SystemExit('S2 geometry repair is opt-in')
    from ..recovery.freeze import verify
    verify()
    GEOM_REPORT.mkdir(parents=True,exist_ok=True);GEOM_RUNS.mkdir(parents=True,exist_ok=True)
    if (GEOM_REPORT/'S2_protocol.json').exists():raise SystemExit('Already prepared; do not overwrite an attempt')
    protocol=read(REC_REPORT/'S2_protocol.json');protocol.update({'version':3,'candidate':'S2_geometry_recovery',
        'bounded_dynamic_scale':True,'static_rgb_weight':.5,
        'loss':'Existing RGB/foreground/coverage + symmetric0.5 conditional-static RGB; scale policy bounded to[0.15,2.5] canonical spacings',
        'repair_trigger':'Read-only5k audit: group0/1 all scales at1e-4 by42; groups2/3 can inflate to0.15; static-region corruption and missing group contributions',
        'comparison_limit':'Scale policy and domain-balanced image objective change together; sequential repair, not a causal single-factor comparison'})
    write(GEOM_REPORT/'S2_protocol.json',protocol)
    receipt=read(REC_REPORT/'S2_data_receipt.json')
    for obj in receipt['objects']:
        assert sha(REC_RUNS/obj['file'])==obj['sha256'];shutil.copyfile(REC_RUNS/obj['file'],GEOM_RUNS/obj['file'])
    receipt['protocol_sha256']=sha(GEOM_REPORT/'S2_protocol.json');receipt['unchanged_initialization_receipt_sha256']=sha(REC_REPORT/'S2_data_receipt.json')
    write(GEOM_REPORT/'S2_data_receipt.json',receipt)


if __name__=='__main__':main()
