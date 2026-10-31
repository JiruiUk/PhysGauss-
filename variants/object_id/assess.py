"""Paired final-test assessment; no ground-truth instance-ID accuracy claim."""
import json
import math
import statistics
from .run import NAMES,RUNS,REPORTS,engine


def assess():
    records={}
    for name in NAMES:
        d=RUNS/name/'rubber_duck'
        records[name]={'metrics':engine.read_json(d/'results.json')['ours_50000'],
            'per_view':engine.read_json(d/'per_view.json')['ours_50000']}
    comparisons=[]
    for name,reference in [(NAMES[1],NAMES[0]),(NAMES[2],NAMES[0]),(NAMES[2],NAMES[1])]:
        a,b=records[name],records[reference]
        changes={k:a['metrics'][k]-b['metrics'][k] for k in engine.METRICS}
        gains=[a['per_view']['PSNR'][key]-b['per_view']['PSNR'][key] for key in a['per_view']['PSNR']]
        comparisons.append({'model':name,'reference':reference,'delta':changes,
            'PSNR_improved_frames':sum(v>0 for v in gains),'PSNR_worse_frames':sum(v<0 for v in gains),
            'PSNR_median_delta':statistics.median(gains),
            'all4_image_metrics_better':all(changes[k]<0 if k=='LPIPS' else changes[k]>0 for k in engine.METRICS)})
    result={'status':'completed','snapshot':'final50000','test_frames':50,'comparisons':comparisons,
        'ground_truth_object_ID_accuracy':'unavailable','semantic_identity_improvement':'not_established_without_instance_evaluation',
        'historical_test_exposure':True,'single_scene_seed':True,'identity_training_uses_instance_labels':False}
    engine.write_json(REPORTS/'assessment.json',result)
    lines=['# Object ID experiment assessment','',
        'Matched C1, frozen identity and learned identity; total50k each, same full3k prefix.',
        'Final50k evaluated on the same50 held-out test images after validation selection was locked.','',
        '| Model minus reference | PSNR delta | SSIM delta | LPIPS delta | MS-SSIM delta | PSNR better frames |',
        '|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append('| '+c['model']+' minus '+c['reference']+' | '+' | '.join(f"{c['delta'][k]:+.7f}" for k in engine.METRICS)+f" | {c['PSNR_improved_frames']}/50 |")
    lines+=['','O2 vs O0 measures the combined object-motion package; O2 vs O1 probes learned membership against frozen geometric membership.',
        'Image changes alone do not establish correct physical object IDs. Check identity renders, occupancy, gradient and shuffled-identity diagnostics.',
        'C1 foreground alpha supervision remains; only the added identity module is instance-label-free.',
        'No nonpenetration or material-identification claim.','',
        'RGB/GT gallery: '+str(RUNS/'render_exports/final50000/gt_c1_frozen_learned/index.html'),
        'Identity maps: '+str(RUNS/'identity_exports/index.html')]
    engine.write_text_atomic(REPORTS/'assessment.md','\n'.join(lines)+'\n')
