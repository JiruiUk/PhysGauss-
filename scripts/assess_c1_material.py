"""Assess completed C1/fixed-E/learned-E runs without changing selections."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs/pidg_c1_material'
REPORTS=ROOT/'experiments/c1_material'
NAMES=['P0_C1_control','P1_fixed_E','P2_learn_E']
METRICS=['PSNR','SSIM','LPIPS','MS-SSIM']
SOURCES={}


def read(path, jsonl=False):
    SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    text=path.read_text(encoding='utf-8')
    return [json.loads(s) for s in text.splitlines() if s.strip()] if jsonl else json.loads(text)


def main():
    assert read(RUNS/'controller.json')['status']=='completed'
    selection=read(RUNS/'selection_lock.json')
    models=[]
    for name in NAMES:
        d=RUNS/name/'rubber_duck'
        results=read(d/'results.json');pv=read(d/'per_view.json')['ours_50000']
        for key in METRICS:
            assert set(pv[key])=={f'{i:05d}.png' for i in range(50)}
            assert all(math.isfinite(v) for v in pv[key].values())
            assert abs(statistics.mean(pv[key].values())-results['ours_50000'][key])<1e-5
        best=read(d/'metric_best.json')
        assert best==selection['validation_selections'][name]
        branch=read(d/'branch.json');assert branch['iteration']==3000
        material=None
        if name!=NAMES[0]:
            rows=read(d/'material_loss.jsonl',True)
            material={'updates_scheduled':len(rows),'E_first':rows[0]['E'],'E_last_logged_before_update':rows[-1]['E'],
                'E_range':[min(r['E'] for r in rows),max(r['E'] for r in rows)],
                'valid_F_fraction_mean':statistics.mean(r['valid_F_fraction'] for r in rows),
                'valid_F_fraction_min':min(r['valid_F_fraction'] for r in rows)}
            for key in ['weighted_loss','F_gradient_norm','predicted_stress_gradient_norm','raw_constitutive']:
                values=[r[key] for r in rows];assert all(math.isfinite(v) for v in values)
                material[key]={'median':statistics.median(values),'max':max(values),
                               'max_record':rows[max(range(len(rows)),key=lambda i:values[i])]}
        models.append({'name':name,'final50k':results['ours_50000'],'coverage':read(d/'coverage_metrics_test.json')['50000']['mean'],
            'validation_selected_PSNR':{'iteration':best['PSNR']['iteration'],'test':results['ours_best']},
            'per_view':pv,'material':material,'branch_sha256':branch['identical_full_checkpoint_sha256']})
    assert len({m['branch_sha256'] for m in models})==1
    comparisons={}
    for a,b in [(1,0),(2,0),(2,1)]:
        out={'mean_delta':{},'per_frame':{}}
        for key in METRICS:
            values=[models[a]['per_view'][key][f'{i:05d}.png']-models[b]['per_view'][key][f'{i:05d}.png'] for i in range(50)]
            out['mean_delta'][key]=models[a]['final50k'][key]-models[b]['final50k'][key]
            out['per_frame'][key]={'median':statistics.median(values),
                'improved_frames':sum(v<0 if key=='LPIPS' else v>0 for v in values),
                'values_by_original_frame':{str(1+4*i):v for i,v in enumerate(values)}}
        comparisons[f'P{a}-P{b}']=out
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':{'iterations':50000,'test_frames':50,
        'split':[120,30,50],'shared_full3k_sha256':models[0]['branch_sha256'],'historical_test_exposure':True,
        'primary_control':'new P0 repeat; not historical C1 score','single_scene_seed':True},
        'models':models,'comparisons':comparisons,'sources':SOURCES,
        'conclusion':'Current global effective-E constitutive formulation degrades image metrics. Learned E shows no consistent recovery versus fixed E.',
        'causal_limits':['finite but large dL/dF spikes are a risk signal, not proven sole cause',
            'E changed only about0.65 percent; this does not establish that optimized material models never help',
            'no true material labels or calibrated physical units; canonical geometry is not a known stress-free reference']}
    (REPORTS/'assessment.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# C1 shared-stiffness assessment','','All3 runs completed. Primary table: final50k, same50 test images.',
        'Matched shared full3k branch; validation-only selections. Historical test exposure persists.','',
        '| Model | PSNR | SSIM | LPIPS | MS-SSIM | Foreground PSNR | Mask IoU | Interior low coverage % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in models:
        s,c=m['final50k'],m['coverage']
        lines.append('| '+' | '.join([m['name'],*[f'{s[k]:.6f}' for k in METRICS],f'{c["foreground_psnr"]:.4f}',
            f'{c["mask_iou"]:.6f}',f'{100*c["interior_hole_rate"]:.6f}'])+' |')
    lines+=['','## Observed effects','',
        'P1-P0 PSNR -2.76335dB;48/50 frames worse. P2-P0 -2.76999dB;49/50 worse.',
        'Foreground PSNR and Mask IoU also decline. Learning E versus fixed E gives mixed small mean differences.',
        'Learned E:1 ->0.993480 before the final update, about0.652% change;470 scheduled material updates.',
        'P2 largest logged weighted constitutive loss13.34475 and ||dL/dF||187195.078 at iteration28400.',
        'P1 maximum ||dL/dF||1378.368 at49000. These are local F derivatives, not model-weight gradient norms.',
        'The nearby recorded validation PSNR does not show an immediate drop across the P2 event; a single spike is not a causal explanation by itself.','',
        '## Judgment','',
        'Keep P0 as the image-quality control. Current global effective-material regularization is not a default improvement.',
        'Before another ablation, inspect F admissibility/conditioning, constitutive-gradient tails and interval compensation; validate rest-configuration and material-region assumptions.',
        'Separate these checks from any later material learning-rate or weight sweep. No new training is launched by this assessment.','',
        '[GT + P0 + P1 + P2](../../runs/pidg_c1_material/render_exports/final50000/gt_p0_p1_p2/index.html)',
        '[Raw run results](results.md) | [Full assessment and source hashes](assessment.json)']
    (REPORTS/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':'completed','assessment':str(REPORTS/'assessment.md'),'PSNR_deltas':{k:v['mean_delta']['PSNR'] for k,v in comparisons.items()}},indent=2))


if __name__=='__main__':main()
