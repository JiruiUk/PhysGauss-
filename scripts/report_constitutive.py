"""Render separate official long and controlled smoke result tables."""
from datetime import datetime
import csv
import io
import sys
from pathlib import Path
from pidg_benchmark import PROJECT,read_json,write_json,write_text_atomic,completed_metrics

RUNS=PROJECT/'runs/pidg_constitutive'
sys.path.insert(0,str(PROJECT))


def matrix():
    import yaml
    return yaml.safe_load((PROJECT/'experiments/pidg_experiment_matrix.yaml').read_text(encoding='utf-8'))


def tag_for(config,mode):return f'm10_{mode}_{config}'


def assert_comparable(candidate,reference):
    for key in ('scene','iterations','resolution_width','seed','profile','smoke_only'):
        if candidate.get(key)!=reference.get(key):raise ValueError(f'Incomparable protocol: {key}')


def rows_for(plan,mode):
    ids=plan['execution']['selected_long_configs' if mode=='long' else 'controlled_smoke_configs']
    rows=[]
    for config_id in ids:
        config=read_json(PROJECT/f'experiments/configs/{config_id}.json');scene=config['scene']
        run=RUNS/tag_for(config_id,mode)/scene
        row={'experiment':config['experiment'],'configuration_id':config_id,'scene':scene,'run':str(run),'status':'queued'}
        if (run/'reproduction.json').exists():
            info=read_json(run/'reproduction.json');row.update(status=info['status'],iterations=info['iterations'],resolution_width=info['resolution_width'])
            if (run/'checkpoints/resume.json').exists():row['checkpoint_iteration']=read_json(run/'checkpoints/resume.json')['iteration']
            if info['status']=='completed':
                try:
                    _,row['metrics']=completed_metrics(run)
                    if mode=='smoke' and (info.get('iterations'),info.get('resolution_width'),info.get('seed'),info.get('smoke_only'),info.get('profile'))!=(300,400,0,True,'controlled_synthetic_smoke'):
                        raise ValueError('Unexpected controlled smoke protocol')
                    if mode=='long':
                        baseline,reference=completed_metrics(PROJECT/'runs/pidg_official'/scene)
                        assert_comparable(info,baseline);row['delta']={key:row['metrics'][key]-reference[key] for key in reference}
                except (ValueError,KeyError,OSError) as error:row.update(status='invalid_results',error=str(error))
            if (run/'extension.jsonl').exists():
                lines=(run/'extension.jsonl').read_text(encoding='utf-8').splitlines()
                try:row['physics']=__import__('json').loads(lines[-1]) if lines else {}
                except ValueError:pass
            if (run/'identifiability.json').exists():
                d=read_json(run/'identifiability.json');row['identifiability']={k:v for k,v in d.items() if k in ('conditional_physical_residual','controlled_reference_errors')}
        if config.get('preset_scope'):row['preset_scope']=config['preset_scope']
        rows.append(row)
    return rows


def report(plan=None):
    plan=matrix() if plan is None else plan
    long=rows_for(plan,'long');smoke=rows_for(plan,'smoke')
    baseline_info,baseline=completed_metrics(PROJECT/'runs/pidg_official/rubber_duck')
    result={'updated_at':datetime.now().astimezone().isoformat(),'selected_scenes':plan['execution']['selected_long_scenes'],
            'long_rows':long,'smoke_rows':smoke,'baseline':baseline,'baseline_reused':True,
            'training_reuse':{'A5_single':'A4_single; post-fit diagnostics only'},
            'unavailable':{'official_multi_material':'known material labels not supplied','dnerf':'data not installed; outside active single-scene scope','hypernerf':'data not installed; outside active single-scene scope'},
            'all_selected_training_completed':all(r['status']=='completed' for r in long)}
    milestone_path=PROJECT/'experiments/milestones.json'
    if milestone_path.exists():
        milestones=read_json(milestone_path);previous=__import__('copy').deepcopy(milestones)
        if 'M10' in milestones and milestones['M10']['status']!='next':
            count=sum(r['status']=='completed' for r in long)
            state='complete' if result['all_selected_training_completed'] else 'failed' if any(r['status'] in ('failed','invalid_results') for r in long) else 'saved_stopped' if any(r['status']=='saved_stopped' for r in long) else 'running' if any(r['status'] in ('training','rendering','evaluating','diagnosing') for r in long) else 'pending'
            milestones['M10'].update(status='passed_automation_long_training_'+state,long_completed=count,long_planned=len(long))
            if milestones!=previous:write_json(milestone_path,milestones)
    write_json(PROJECT/'experiments/results.json',result)
    text=f"# PIDG constitutive plan results\n\nUpdated: {result['updated_at']}\n\nM0 reused five completed original baselines. Active long scope: Rubber Duck only, 50k steps, width1600, seed0. Short controlled data are reported separately.\n\n"
    text+='## Table 1: selected official compatibility benchmark\n\n| Method | Status | Checkpoint | PSNR | Delta PSNR | SSIM | LPIPS |\n|---|---|---:|---:|---:|---:|---:|\n'
    text+=f"| A0 original (reused) | completed | 50000 | {baseline['PSNR']:.5f} | 0 | {baseline['SSIM']:.5f} | {baseline['LPIPS']:.5f} |\n"
    def fmt(value):return f'{value:.6f}' if isinstance(value,(int,float)) else '—'
    for row in long:
        metrics=row.get('metrics',{});delta=row.get('delta',{})
        text+=f"| {row['configuration_id']} | {row['status']} | {row.get('checkpoint_iteration','—')} | {fmt(metrics.get('PSNR'))} | {fmt(delta.get('PSNR'))} | {fmt(metrics.get('SSIM'))} | {fmt(metrics.get('LPIPS'))} |\n"
    text+='\nA3 official multi-material comparison awaits known labels. A4_single/A6_single use a single shared material; they are explicitly compatible subsets, not real multi-material results. A5 diagnostics reuse A4 training. Full multi-material/interface modules are validated on manufactured data below. D-NeRF/HyperNeRF metrics remain unavailable, not zero. Historical A0 runtime hashes were not recorded.\n'
    text+='\n## Table 2: controlled engineering ablation\n\nAll rows below use manufactured two-region data, 300 steps, width400, seed0; these are not 50k official benchmark gains.\n\n| Method | Status | PSNR | SSIM | LPIPS | CMR sampled loss | Constitutive raw | Invalid F | Interface proxy |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for row in smoke:
        m=row.get('metrics',{});p=row.get('physics',{})
        text+=f"| {row['configuration_id']} | {row['status']} | {fmt(m.get('PSNR'))} | {fmt(m.get('SSIM'))} | {fmt(m.get('LPIPS'))} | {fmt(p.get('cauchy_residual_legacy'))} | {fmt(p.get('loss_constitutive_raw'))} | {fmt(p.get('invalid_F_fraction'))} | {fmt(p.get('loss_interface_raw'))} |\n"
    text+='\nCMR retains the released sample scaling and time convention. Interface values use a canonical centre-line normal proxy. New physical parameters are not claimed recovered. Future-time images use the test-based snapshot protocol, so no leakage-free forecasting claim is made.\n'
    from pidg_ext.identifiability import ambiguity_experiment
    ambiguity=ambiguity_experiment()
    text+='\n## Table 3: manufactured stiffness/force ambiguity\n\nAnalytic solutions to a manufactured balance; not material recovery from official images.\n\n| Case | Data rank | Objective rank | Condition | Min SV | mu error | Force error | Max correlation |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for name,row in ambiguity['cases'].items():
        text+=f"| {name} | {row['data_effective_rank']} | {row['effective_rank']} | {row['condition_number']} | {fmt(row['min_singular_value'])} | {fmt(row['mu_relative_error'])} | {fmt(row['force_relative_error'])} | {fmt(row.get('max_material_force_correlation'))} |\n"
    text+='\nA prior can improve objective conditioning while the data remain ambiguous. Actual post-fit Jacobians condition on fixed neural fields and are not proof of global identifiability. The earlier controlled A5 smoke retained about 20% material error and 100% force error despite full conditional rank.\n'
    write_text_atomic(PROJECT/'experiments/results.md',text)
    fields=next(csv.reader((PROJECT/'experiments/RESULTS_TEMPLATE.csv').read_text().splitlines()))
    output=io.StringIO();writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
    writer.writerow({'experiment_id':'A0','suite':'official_long','scene':'rubber_duck','seed':0,'psnr':baseline['PSNR'],'ssim':baseline['SSIM'],
                     'ms_ssim':baseline['MS-SSIM'],'lpips':baseline['LPIPS'],'notes':'Completed historical 50k baseline reused; historical runtime hashes unavailable'})
    for group,rows in [('official_long',long),('controlled_smoke',smoke)]:
        for row in rows:
            m=row.get('metrics',{});p=row.get('physics',{});d=row.get('identifiability',{}).get('conditional_physical_residual',{})
            errors=row.get('identifiability',{}).get('controlled_reference_errors') or {}
            material_errors=(errors.get('mu_relative_error') or [])+(errors.get('kappa_relative_error') or [])
            writer.writerow({'experiment_id':row['experiment'],'suite':group,'scene':row['scene'],'seed':0,
                             'psnr':m.get('PSNR'),'ssim':m.get('SSIM'),'ms_ssim':m.get('MS-SSIM'),'lpips':m.get('LPIPS'),
                             'cauchy_residual':p.get('cauchy_residual_legacy'),'constitutive_residual':p.get('loss_constitutive_raw'),
                             'invalid_F_fraction':p.get('invalid_F_fraction'),'jacobian_min_sv':d.get('min_singular_value'),
                             'material_param_error':sum(material_errors)/len(material_errors) if material_errors else None,
                             'force_error':errors.get('force_relative_l2_error'),
                             'jacobian_max_sv':d.get('max_singular_value'),'jacobian_condition_number':d.get('condition_number'),
                             'stiffness_force_corr':d.get('max_material_force_correlation'),'notes':row['status']+'; '+row.get('preset_scope','')})
    write_text_atomic(PROJECT/'experiments/results.csv',output.getvalue())
    return result


if __name__=='__main__':report()
