"""Pure, bounded decisions from VALIDATION screens; never accepts test metrics."""
from copy import deepcopy
from variants.integrity_tuning.runtime import quality_guard


def propose(screens,configs,reference_metrics,reference_geometry):
    safe=[r for r in screens if quality_guard(r['metrics'],reference_metrics) and
        r['integrity']['background_false_coverage']<=reference_geometry['background_false_coverage']+.005]
    pool=safe or screens
    if not pool:return {'action':'stop','reasons':['No completed validation screen is available.']}
    best=min(pool,key=lambda r:(r['integrity']['integrity_score'],r['integrity']['p95_score']))
    config=configs[best['model']];geometry=best['integrity'];reasons=[]
    knobs={key:deepcopy(config[key]) for key in ('integrity_coverage_weight','integrity_strain_weight','coverage_density')}
    knobs.update(coverage_add_limit=128,integrity_tail_weight=0.,integrity_tail_start=8000,
        integrity_tail_interval=4,integrity_tail_fraction=.2,integrity_tail_alpha_min=.8)
    reasons.append('Start from the best eligible20k validation configuration: '+best['model']+'.')
    if not safe:
        knobs['integrity_coverage_weight']*=.25;knobs['integrity_strain_weight']*=.25;knobs['coverage_density']=False
        reasons.append('All screens failed image/spill safeguards: quarter the new regularizers and disable cloning.')
    if knobs['coverage_density']:
        reasons.append('Density selected by validation: halve its per-window clone cap to128 as a conservative ablation.')
    # A high-alpha appearance defect must be addressed through RGB, not by
    # increasing alpha until a hole metric becomes zero.
    appearance_dominates=reference_geometry['opaque_appearance_error']>5*max(reference_geometry['core_low_50'],1e-6)
    if appearance_dominates or geometry['opaque_appearance_error']>reference_geometry['opaque_appearance_error']+.002:
        knobs['integrity_tail_weight']=.01
        reasons.append('Opaque appearance errors dominate alpha holes: add0.01 worst20%-foreground RGB emphasis.')
    if geometry['background_false_coverage']>reference_geometry['background_false_coverage']+.002 and knobs['integrity_coverage_weight']:
        knobs['integrity_coverage_weight']*=.5
        reasons.append('Validation spill is elevated: halve interior-coverage pressure.')
    if not any((knobs['integrity_coverage_weight'],knobs['integrity_strain_weight'],knobs['coverage_density'],knobs['integrity_tail_weight'])):
        return {'action':'stop','reasons':reasons+['No evidence-backed change remains; do not retrain the baseline.']}
    if not knobs['integrity_tail_weight'] and not knobs['coverage_density'] and \
        knobs['integrity_coverage_weight']==config['integrity_coverage_weight'] and knobs['integrity_strain_weight']==config['integrity_strain_weight']:
        return {'action':'stop','reasons':reasons+['No active module differs from the existing screen; skip duplicate training.']}
    return {'action':'train','source_model':best['model'],'knobs':knobs,'reasons':reasons,
        'evidence_split':'validation','source_metrics':best['metrics'],'source_geometry':geometry,
        'safe_screens':[r['model'] for r in safe],'maximum_new_long_runs':1}


def review_final(proposal,final_validation,reference_metrics,reference_geometry):
    """Recheck the completed source model before freezing the new recipe."""
    result=deepcopy(proposal)
    if result['action']!='train' or final_validation is None:return result
    g=final_validation['integrity'];metrics=final_validation['metrics'];knobs=result['knobs']
    if not quality_guard(metrics,reference_metrics) or g['background_false_coverage']>reference_geometry['background_false_coverage']+.002:
        knobs['integrity_coverage_weight']*=.5;knobs['integrity_strain_weight']*=.5;knobs['coverage_density']=False
        result['reasons'].append('Completed50k validation failed image/spill safeguards: weaken regularization and disable late cloning before the feedback run.')
    if g['opaque_appearance_error']>reference_geometry['opaque_appearance_error']+.002:
        knobs['integrity_tail_weight']=max(.01,knobs['integrity_tail_weight'])
        result['reasons'].append('Completed50k validation still has elevated opaque RGB errors; retain explicit RGB-tail correction.')
    result['completed_source_validation']=final_validation
    return result


def retry(proposal,outcome,reference_metrics,reference_geometry):
    result=deepcopy(proposal);knobs=result['knobs'];g=outcome['integrity'];old=proposal['source_geometry']
    reasons=[]
    if not quality_guard(outcome['metrics'],reference_metrics) or g['background_false_coverage']>reference_geometry['background_false_coverage']+.005:
        for key in ('integrity_coverage_weight','integrity_strain_weight','integrity_tail_weight'):knobs[key]*=.5
        knobs['coverage_density']=False
        reasons.append('Follow-up20k failed quality/spill checks: halve auxiliary weights and disable cloning in a fresh branch.')
    elif g['opaque_appearance_error']>old['opaque_appearance_error']+.002:
        knobs['integrity_tail_weight']=min(.02,knobs['integrity_tail_weight']+.005)
        knobs['integrity_coverage_weight']*=.5
        reasons.append('Opaque RGB errors increased while image guard passed: strengthen RGB-tail emphasis and weaken alpha pressure.')
    elif g['fragment_loss']>old['fragment_loss']*1.05 and knobs['integrity_strain_weight']:
        knobs['integrity_strain_weight']*=.5
        reasons.append('Fragmentation increased with strain active: weaken strain for a controlled follow-up, without assuming causation.')
    elif g['core_low_90']>old['core_low_90']*1.05 and g['background_false_coverage']<=reference_geometry['background_false_coverage']+.002:
        knobs['integrity_coverage_weight']=min(.002,max(.0005,knobs['integrity_coverage_weight']*1.5))
        reasons.append('Coverage is still deficient with stable spill: modestly increase coverage weight.')
    else:return {'action':'stop','reasons':['No sufficiently clear validation-directed retry; retain existing models.']}
    if not any((knobs['integrity_coverage_weight'],knobs['integrity_strain_weight'],knobs['coverage_density'],knobs['integrity_tail_weight'])):
        return {'action':'stop','reasons':['Proposed retry would duplicate the baseline; skip it.']}
    result.update(reasons=reasons,previous_outcome=outcome,evidence_split='validation')
    return result
