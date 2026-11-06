"""Independent accounting of pair census, metadata, denominators and receipts."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
REPORT=BASE/'v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()

def main():
    p=read(BASE/'protocol.json');meta=read(ROOT/p['metadata']);frames=meta['frames']
    expected=set()
    for a in p['anchors']:
        for c in range(120):expected.add((c,a))
        for dc in p['offsets']:
            for dt in p['offsets']:expected.add((a+dc,a+dt))
    assert len(expected)==482
    times={float(f['time']) for f in frames};poses={tuple(np.asarray(f['transform_matrix']).ravel()) for f in frames}
    assert len(times)==len(poses)==len(frames)==120
    results={}
    for name,item in p['models'].items():
        assert sha(ROOT/item['path'])==item['sha256']
        q=read(REPORT/(name+'.json'));actual={(r['camera_index'],r['time_index']) for r in q['rows']}
        assert actual==expected and len(q['rows'])==len(expected)
        assert q['state_sha256_before']==q['state_sha256_after']
        observed=0;undefined=0;clipped=0
        for row in q['rows']:
            c,t=row['camera_index'],row['time_index'];d=row['dynamic']
            assert d['pixels']==800*450 and 0<=d['dark_high_alpha_pixels']<=d['alpha_gt05_pixels']<=d['pixels']
            assert (row['observed_PSNR'] is not None)==(c==t)
            observed+=c==t;undefined+=d['dark_fraction_on_alpha'] is None
            clipped+=d['shape_proxies']['touches_image_border']
            if d['alpha_gt05_pixels']:assert d['dark_fraction_on_alpha']==d['dark_high_alpha_pixels']/d['alpha_gt05_pixels']
        for a in p['anchors']:
            group=[r for r in q['rows'] if r['time_index']==a]
            bins=q['summary'][str(a)]['angle_bins'];assert sum(b['views'] for b in bins)==120
            for b in bins:
                lo,hi=b['degrees'];chosen=[]
                for row in group:
                    r1=np.asarray(frames[a]['transform_matrix'])[:3,:3];r2=np.asarray(frames[row['camera_index']]['transform_matrix'])[:3,:3]
                    u,_,v=np.linalg.svd(r1);r1=u@v;u,_,v=np.linalg.svd(r2);r2=u@v
                    theta=float(np.degrees(np.arccos(np.clip(np.dot(r1[:,2],r2[:,2]),-1,1))))
                    assert abs(theta-row['camera_time_distance']['view_direction_deg'])<1e-10
                    if lo<=theta<hi:chosen.append(row['dynamic'])
                valid=[v['dark_fraction_on_alpha'] for v in chosen if v['dark_fraction_on_alpha'] is not None]
                assert len(chosen)==b['views'] and len(valid)==b['valid_dark_ratios']
                assert b['dark_fraction_median']==(None if not valid else float(np.median(valid)))
                assert b['undefined_dark_ratios']==len(chosen)-len(valid)
        for path,h in q['output_files'].items():assert sha(ROOT/path)==h,path
        assert len(q['native1600'])==21
        results[name]=dict(pairs=482,observed_diagonal_pairs=observed,undefined_ratios=undefined,
            clipped_pairs=clipped,output_PNGs=len(q['output_files']),native1600_combinations=21,
            source_checkpoint_unchanged=True,full_state_unchanged=True,
            full_second_process_native_replay=read(REPORT/'independent_audit.json')['models'][name]['passed'])
    for name,h in read(REPORT/'source_freeze.json')['files'].items():assert sha(ROOT/name)==h,name
    result=dict(passed=True,models=results,training_images=120,unique_times=120,unique_poses=120,
        same_time_multicamera_observations=0,training_updates=0,independent_novel_view_GT=False,
        independent_validator_sha256=sha(Path(__file__)))
    (REPORT/'final_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
