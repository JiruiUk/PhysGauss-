"""Independent artifact checks; does not load a GPU or rerun model training."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/surface_support';REPORT=ROOT/'experiments/surface_support'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    lock=read(REPORT/'input_lock.json');assert all(sha(ROOT/p)==v for p,v in lock.items())
    expected=[int(Path(f['file_path']).stem) for f in read(ROOT/'data/derived/kinematic_resplit/training/transforms_train.json')['frames']]
    assert len(expected)==120
    evidence={};artifacts={}
    for name in ['C1_final','P1_16k']:
        m=read(REPORT/(name+'.json'));rows=m['frames'];assert [r['frame'] for r in rows]==expected
        a=np.load(OUT/name/'trajectories.npz');assert np.array_equal(a['frames'],expected)
        assert a['centers'].shape==(120,sum(m['early_partial_patch_counts']),3)
        assert a['first20_visibility_lower_bound'].shape==a['centers'].shape[:2]
        assert np.isfinite(a['centers']).all() and np.isfinite(a['projected_footprint']).all()
        assert len(set(a['late86_probe_ids'].tolist())&set(a['kernel_ids'].tolist()))==0
        for r in rows:
            assert 0<=r['exact_known_share']<=1.0001 and 0<=r['first20_total_share']<=1.0001
            assert r['first20_known_share_true_alpha']<=r['exact_known_share']+2e-4
            assert abs(sum(r['exact_group_share'])-r['exact_known_share'])<2e-6
            p=OUT/name/f"frame{r['frame']:04d}.png"
            with Image.open(p) as image:assert image.size==(2000,265);image.verify()
            artifacts[str(p.relative_to(ROOT))]=sha(p)
        for p in (OUT/name).glob('maps*.npz'):
            z=np.load(p);assert np.all(z['actual']<=z['isolated']+2e-4)
            assert np.all(z['actual']<=z['true_alpha']+2e-4)
            assert all(np.isfinite(z[k]).all() for k in z.files)
        for p in (OUT/name).glob('*.npz'):artifacts[str(p.relative_to(ROOT))]=sha(p)
        zeros=[r['frame'] for r in rows if r['exact_known_share']==0]
        evidence[name]={'frames':len(rows),'zero_full_compositor_known_frames':len(zeros),'first_zero':zeros[0],
          'minimum_first20_metadata_share':min(r['first20_total_share'] for r in rows),'maximum_first20_metadata_share':max(r['first20_total_share'] for r in rows)}
    z=np.load(OUT/'image_tracks.npz');labels=z['labels']
    assert labels.shape==(120,225,400) and set(np.unique(labels))<=set([-1,0,1,2,3])
    for k in range(4):
        lost=np.where((labels==k).sum((1,2))==0)[0]
        if len(lost):assert not (labels[lost[0]:]==k).any()
    contracts=read(REPORT/'renderer_contracts.json');assert contracts['passed']
    for r in contracts['binary_mask_training_frames']:
        assert r['values']==[0,255] and sha(ROOT/f"data/PIDG/rubber_duck/mask/{r['frame']:04d}.png")==r['sha256']
    assert all(sha(ROOT/p)==v for p,v in contracts['source_sha256'].items())
    for rel in ['comparison.png','center_review.png','center_trajectories_3d.png','sequence_review.png','review_C1.png','review_P1.png','correspondence_and_metadata.png','index.html','image_tracks.npz']:
        artifacts[str((OUT/rel).relative_to(ROOT))]=sha(OUT/rel)
    manual=read(REPORT/'manual_review.json') if (REPORT/'manual_review.json').exists() else None
    if manual:
        assert all(sha(ROOT/p)==v for p,v in manual['reviewed_artifact_sha256'].items())
    report={'passed':True,'all_input_sha256_unchanged':True,'full_train_frames_each':120,'unknown_not_reassigned':True,
      'no_reappearance_after_track_loss':True,'full_vs_first20_checks_passed':True,'isolated_cohort_pixelwise_dominates_actual':True,
      'renderer_contracts_passed':True,'all_120_masks_binary_not_instance_ID':True,
      'models':evidence,'manual_review_pending':manual is None,'artifacts_sha256':artifacts}
    (REPORT/'independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
