"""Report predeclared metrics, preserve negative results and request visual review."""
from .run import ROOT, REPORT, NAMES, engine, plan


def main():
    metrics={n:{s:engine.read_json(REPORT/f'{n}_{s}.json')['metrics'] for s in ['val','test']} for n in NAMES}
    statics=engine.read_json(REPORT/'static_diagnostics.json')
    support=engine.read_json(REPORT/'static_visible_support.json')
    shape=engine.read_json(REPORT/'shape_summary.json')
    a,b=NAMES;delta={s:{k:metrics[b][s][k]-metrics[a][s][k] for k in metrics[a][s]} for s in ['val','test']}
    guard=plan()['image_guardrails']
    gates={'validation_PSNR':delta['val']['PSNR']>=-guard['validation_PSNR_max_drop_dB'],
           'validation_standard_LPIPS':metrics[b]['val']['LPIPS']<=metrics[a]['val']['LPIPS']*(1+guard['validation_standard_VGG_LPIPS_max_relative_increase'])}
    result={'models':metrics,'H1_minus_H0':delta,'image_guardrails':gates,'static':statics,'support':support,'shape':shape,
        'status':'evaluated_pending_manual_review','promotion_decided':False,'novel_view_GT':False,
        'source_and_protocol':plan(),'conclusion':'Numeric evaluation complete. Structural static invariance is not proof of correct 3D geometry. Native visual review and independent-evidence limits must determine the next experiment.'}
    engine.write_json(REPORT/'numeric_summary.json',result)
    lines=['# R0 static-pool evaluation — pending native visual review','',
        'Two arms from identical full15k state to final50k;35000 actual updates each; no new transfer, shape, contact or constitutive loss.',
        'The treatment includes a fixed rendering map for the accepted static rows and exclusion of those rows from dynamic CMR.',
        'Historical S0 is a contextual reference, not the matched control.','',
        '| Model | Validation PSNR | Test PSNR | Test SSIM | Standard VGG LPIPS |',
        '|---|---:|---:|---:|---:|']
    for n in NAMES:
        v,t=metrics[n]['val'],metrics[n]['test']
        lines.append(f"| {n} | {v['PSNR']:.6f} | {t['PSNR']:.6f} | {t['SSIM']:.8f} | {t['LPIPS']:.8f} |")
    lines+=['',f'Predeclared image guardrails: {gates}. Passing them alone is not a successful geometry result.','',
        '| Diagnostic | H0 | H1 |','|---|---:|---:|']
    for key in ['max_center_time_difference_RMS','max_covariance_time_difference_RMS',
                'static_image_track_reprojection_mean_px_400','triangulation_held_observation_mean_px_400']:
        lines.append(f'| {key} | {statics[a][key]:.8g} | {statics[b][key]:.8g} |')
    for split in ['train','val']:
        k='mean_static_share_of_blue_alpha'
        lines.append(f"| {split} selected-static blue contribution share | {support[a]['groups'][split][k]:.6f} | {support[b]['groups'][split][k]:.6f} |")
    lines+=['','Static image tracks come from original training images; their odd observations were held only from triangulation, not from model RGB training.',
        'Color subsets and fixed-view morphology are proxies. Novel camera/time image GT and complete physical IDs are unavailable.',
        'Inspect the fixed86 and fixed188 native crops, static support/opacity, both object regions and the moving-camera control before promotion.',
        'If static geometry improves while dynamic fragmentation persists, proceed to an object-support audit; otherwise audit selection/reference geometry first.',
        'Do not launch a weight sweep, contact force model or another full candidate automatically.','',
        '[Video and native image gallery](../../output/static_pool/index.html)']
    (REPORT/'assessment.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (REPORT/'progress.md').write_text('# R0 progress\n\nNumeric evaluation and export completed; native manual review is required.\n\n'
        f'Image guardrails: {gates}. No model has been promoted.\n',encoding='utf-8')
    adaptive=ROOT/'experiments/adaptive_reconstruction_plan.md'
    text=adaptive.read_text(encoding='utf-8')
    marker='\n## R0 completed measurements\n'
    if marker in text:text=text.split(marker)[0]
    text+=marker+'\nFinal50k measurements and videos are available in `static_pool/assessment.md`. '
    text+=f'Image guardrails: {gates}. Native visual review and the next-step decision remain pending. '
    text+='No subsequent full candidate is automatically authorized by passing a scalar proxy.\n'
    adaptive.write_text(text,encoding='utf-8')
    print('Numeric summary, assessment and adaptive-plan status updated; manual review pending.',flush=True)


if __name__=='__main__':main()
