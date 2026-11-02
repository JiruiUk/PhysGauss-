"""Independent CPU recheck of frozen rows, raw-flow motion and full footprint gates."""
from pathlib import Path
import sys,collections
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from output.surface_correspondence.common import *
from output.surface_correspondence.finalize import point_inside
from variants.flow_comparison.geometry import sample,forward_backward_valid


def main():
    torch.set_num_threads(4);lock=read(REPORT/'input_lock.json')
    assert all(sha(ROOT/p)==v for p,v in lock.items())
    contracts=read(REPORT/'renderer_contracts.json');assert contracts['passed']
    assert all(sha(ROOT/p)==v for p,v in contracts['source_sha256'].items())
    result=read(REPORT/'results.json');summary=read(REPORT/'summary.json')
    def arrays(path):
        with np.load(path) as value:return {k:value[k] for k in value.files}
    z=arrays(OUT/'surface_trajectories.npz');image=arrays(ROOT/'output/surface_support/image_tracks.npz')
    tracks=arrays(OUT/'fragment_tracks.npz');out=arrays(OUT/'surface_assignments.npz')
    assert np.array_equal(z['frames'],image['frames']) and len(z['frames'])==120
    assert np.array_equal(z['frames'],out['frames']) and np.array_equal(z['kernel_ids'],out['kernel_ids'])
    assert len(np.unique(z['kernel_ids']))==len(z['kernel_ids'])
    assert all(np.isfinite(z[k]).all() for k in ['canonical','centers','projected','footprint_sqrt_det','mass','depth'])
    assert np.min(z['mass'])>=0 and np.max(z['mass'][:,:,:4].sum(-1)-z['mass'][:,:,4])<.005
    manifest=read(FLOW/'manifest.json');pairs={}
    for p in manifest['pairs']:
        with np.load(FLOW/p['file']) as f:pairs[(p['source'],p['target'])]=(torch.tensor(f['forward'],dtype=torch.float32),torch.tensor(f['backward'],dtype=torch.float32))
    # Independent CPU observation sampling for ALL canonical union rows/times.
    flow_max_error=0.
    for i in range(1,120):
        p=torch.tensor(z['projected'][i-1]);q=torch.tensor(z['projected'][i]);f,b=pairs[(int(z['frames'][i-1]),int(z['frames'][i]))]
        sampled=sample(f,p[:,None])[:,0];valid=sample(forward_backward_valid(f,b).float(),p[:,None])[:,0]>.99
        valid&=(p[:,0]>=0)&(p[:,0]<=399)&(p[:,1]>=0)&(p[:,1]<=224)
        expected=torch.linalg.vector_norm(q-p-sampled,dim=-1).numpy();cached=z['raw_flow_error'][i]
        assert np.array_equal(np.isfinite(cached),valid.numpy())
        if valid.any():flow_max_error=max(flow_max_error,float(np.max(np.abs(cached[valid]-expected[valid]))))
    assert flow_max_error<.002
    positions={int(k):i for i,k in enumerate(z['kernel_ids'])};local={};per_bank={}
    bindings=read(REPORT/'bank_bindings.json')
    assert sha(ROOT/bindings['full_checkpoint'])==bindings['full_checkpoint_sha256']
    assert sha(TRAIN/'transforms_train.json')==bindings['training_metadata_sha256']
    assert sha(ROOT/bindings['canonical_particle_union_file'])==bindings['canonical_particle_union_sha256']
    assert all(sha(ROOT/p)==v for p,v in bindings['source_sha256'].items())
    # Once visible fragment evidence terminates, that anchor cannot revive it.
    for a,fragment in enumerate(read(REPORT/'image_fragments.json')):
        start=fragment['start_index']
        for k in range(4):
            column=tracks['component_mapping'][a,start:,k];bad=np.flatnonzero(column<0)
            if len(bad):assert (column[bad[0]:]<0).all()
        assert set(np.unique(tracks['labels'][a]).tolist())<=set([-1,0,1,2,3])
        assert (tracks['labels'][a][image['components']<0]==-1).all()
        for i in range(start,120):
            mapping=tracks['component_mapping'][a,i];positive=mapping[mapping>=0]
            assert len(positive)==len(set(positive.tolist()))
            for k,c in enumerate(mapping):
                if c<0:continue
                region=image['components'][i]==c;labels=tracks['labels'][a,i][region];known=labels[labels>=0]
                assert len(known)>=30 and len(known)>=.25*region.sum()
                assert np.mean(known==k)>=.95
    for bank in result['banks']:
        path=ROOT/bank['bank_file'];assert sha(path)==bank['bank_sha256']
        assert bindings['banks'][bank['name']]['sha256']==bank['bank_sha256']
        with np.load(path) as b:
            ids=b['kernel_ids'];ix=np.array([positions[int(k)] for k in ids]);local[bank['name']]=ix
            assert np.array_equal(b['canonical'],z['canonical'][ix]) and len(ids)==bank['particle_count']>=20
            masks=[];previous=None
            for offset,number in enumerate(bank['seed_frames']):
                i=int(np.where(z['frames']==number)[0][0]);component=tracks['component_mapping'][bank['anchor_index'],i,bank['local_seed']]
                value=np.load(CACHE/f'seed{number:04d}.npz');own=np.zeros(len(value['mass'])) if component<0 else value['mass'][:,component]
                from output.surface_correspondence.audit import center_inside
                inside=np.zeros(len(own),bool) if component<0 else center_inside(value['means'],value['depth'],value['cov'],image['components'][i]==component)
                valid=(own>=.02)&(own/np.maximum(value['mass'][:,4],1e-20)>=.95)&inside
                if previous is not None:
                    p=torch.tensor(previous);q=torch.tensor(value['means']);f,b=pairs[(int(z['frames'][i-1]),int(number))]
                    vf=sample(forward_backward_valid(f,b).float(),p[:,None])[:,0]>.99
                    vf&=(p[:,0]>=0)&(p[:,0]<=399)&(p[:,1]>=0)&(p[:,1]<=224)
                    e=torch.linalg.vector_norm(q-p-sample(f,p[:,None])[:,0],dim=-1)
                    valid&=vf.numpy()&(e.numpy()<=2.)
                masks.append(valid);previous=value['means'].copy()
            assert np.array_equal(np.flatnonzero(np.logical_and.reduce(masks)),ids)
            per_bank[bank['name']]={'future_visible':0,'accepted':0,'reasons':collections.Counter()}
    # Recompute every future cohort gate and every per-point assignment, with
    # explicit arithmetic rather than reusing assess_cohort/particle_support.
    counts=0;coverage=[];assigned_expected=np.full(out['component_assignment'].shape,-1,dtype=np.int8)
    for i,row in enumerate(result['frames']):
        proposals=[set() for _ in z['kernel_ids']]
        for r in row['banks']:
            ix=local[r['bank']];c=r['component'];own=np.zeros(len(ix)) if c<0 else z['mass'][i,ix,c];total=z['mass'][i,ix,4]
            inside=np.zeros(len(ix),bool) if c<0 else point_inside(z['projected'][i,ix],z['depth'][i,ix],z['footprint_sqrt_det'][i,ix],image['components'][i]==c)
            error=z['raw_flow_error'][i,ix];mass=float(own.sum());tm=float(total.sum())
            precision=mass/tm if tm>1e-12 else None;center=float((total*inside).sum()/tm) if tm>1e-12 else None
            flow=float((own*(np.isfinite(error)&(error<=2))).sum()/mass) if mass>1e-12 else None
            accepted=r['phase']=='forward_validation' and c>=0 and mass>=.2 and precision is not None and precision>=.95 and center is not None and center>=.9 and flow is not None and flow>=.9
            assert accepted==r['accepted']
            assert abs(mass-r['own_mass'])<1e-6 and abs(float(inside.mean())-r['center_inside_unweighted_fraction'])<1e-10
            if r['phase']=='forward_validation':
                per_bank[r['bank']]['future_visible']+=int(c>=0);per_bank[r['bank']]['accepted']+=int(accepted)
                per_bank[r['bank']]['reasons'].update(r['unknown_reasons'])
            if accepted:
                good=(own>=.02)&(own/np.maximum(total,1e-20)>=.95)&inside&np.isfinite(error)&(error<=2)
                bank=next(b for b in result['banks'] if b['name']==r['bank'])
                previous_component=tracks['component_mapping'][bank['anchor_index'],i-1,bank['local_seed']] if i>0 else -1
                if previous_component<0:good[:]=False
                else:
                    previous_inside=point_inside(z['projected'][i-1,ix],z['depth'][i-1,ix],z['footprint_sqrt_det'][i-1,ix],image['components'][i-1]==previous_component)
                    source_own=z['mass'][i-1,ix,previous_component];source_total=z['mass'][i-1,ix,4]
                    good&=(source_own>=.02)&(source_own/np.maximum(source_total,1e-20)>=.95)&previous_inside
                counts+=int(good.any())
                for k in ix[good]:proposals[k].add(c)
        for k,values in enumerate(proposals):
            if len(values)==1:assigned_expected[i,k]=next(iter(values))
        ids=np.where(assigned_expected[i]>=0)[0];mass=float(z['mass'][i,ids,assigned_expected[i,ids]].sum())
        assert abs(mass-summary['frames'][i]['accepted_fragment_alpha_mass'])<1e-6
        coverage.append(mass/row['all_duck_alpha_mass'])
    assert np.array_equal(assigned_expected,out['component_assignment'])
    initial=REPORT/'initial_target_only'
    with np.load(initial/'surface_assignments.npz') as previous:
        old_assignment=previous['component_assignment'];assert np.array_equal(previous['kernel_ids'],z['kernel_ids'])
    assert np.array_equal(old_assignment[assigned_expected>=0],assigned_expected[assigned_expected>=0])
    rejected_source_events=int(((old_assignment>=0)&(assigned_expected<0)).sum())
    initial_visual=read(initial/'visual_contracts.json')
    old_key=next(k for k in initial_visual['source_sha256'] if k.replace('\\','/')=='output/surface_correspondence/finalize.py')
    assert sha(initial/'finalize.py')==initial_visual['source_sha256'][old_key]
    assert counts==summary['accepted_forward_bank_frame_events']
    assert abs(np.mean(coverage)-summary['all120_mean_accepted_alpha_coverage'])<1e-10
    assert all(sha(ROOT/p)==v for p,v in result['cache_receipts'].items())
    visuals=read(REPORT/'visual_contracts.json');assert visuals['passed']
    assert all(sha(ROOT/p)==v for p,v in visuals['source_sha256'].items())
    for row in visuals['full_compositor_visuals']:
        assert row['current_surface_integral_relative_error']<.002 and row['early_cohort_integral_relative_error']<.002
        with Image.open(OUT/'review'/f"frame{row['frame']:04d}.png") as im:im.load();assert im.size==(2400,265)
    from output.c1_flow_ablation.reuse_control import control_hashes
    assert control_hashes()==read(ROOT/'experiments/c1_flow_ablation/reuse_control_plan.json')['historical_control_sha256']
    manual=read(REPORT/'manual_review.json');assert manual['review_complete']
    files={str(p.relative_to(ROOT)):sha(p) for base in [REPORT,OUT] for p in base.rglob('*') if p.is_file()
        and p.suffix in ['.json','.npz','.png','.py','.html','.md'] and p.name!='independent_verification.json'}
    write(REPORT/'independent_verification.json',{'passed':True,'locked_input_files':len(lock),'all_SHA_unchanged':True,
        'historical_C1_protection_passed':True,'full_C1_checkpoint_snapshot_match':True,'frozen_bank_seed_intersections_independently_rechecked':len(result['banks']),
        'all120_center_flow_rechecked_on_CPU':True,'maximum_CPU_vs_GPU_flow_error_pixels':flow_max_error,
        'all_cohort_gates_and_particle_unknowns_recalculated':True,'unique_canonical_rows':len(z['kernel_ids']),
        'image_fragment_termination_and_qualification_checked':True,'bank_rows_bound_to_exact_C1_full_checkpoint':True,
        'same_local_fragment_source_AND_target_particle_support_checked':True,
        'source_unobservable_point_frame_events_removed':rejected_source_events,'initial_target_only_source_archive_SHA_verified':True,
        'accepted_forward_bank_frames':counts,'mean_visible_duck_alpha_coverage':float(np.mean(coverage)),
        'per_bank_diagnostics':per_bank,'compositor_visual_frames':[r['frame'] for r in visuals['full_compositor_visuals']],
        'representative_manual_review_completed':True,'new_Gaussian_optimizer_updates':0,'full_reconstruction_integration_approved':False,
        'physical_material_identity_accuracy':None,'no_cross_anchor_physical_identity_merge':True,'artifact_sha256':files})
    print('INDEPENDENT SURFACE VERIFICATION PASSED',counts,len(z['kernel_ids']),np.mean(coverage),flush=True)


if __name__=='__main__':main()
