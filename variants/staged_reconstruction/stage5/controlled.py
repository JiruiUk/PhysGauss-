"""Manufactured loss/optimizer validation, not a PIDG reconstruction benchmark."""
import argparse
import copy
import numpy as np
import torch
from ..prepare_data import REPORT,RUNS,write,sha
from .nonpenetration import nonpenetration


def fit(observed,eligible,weight,steps,initial=None):
    parameter=torch.nn.Parameter(observed.clone());optimizer=torch.optim.Adam([parameter],lr=.02)
    if initial is not None:
        parameter.data.copy_(initial['position']);optimizer.load_state_dict(initial['optimizer'])
    history=[];middle=None
    for step in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        data=(parameter-observed).square().mean()
        contact,info=nonpenetration(parameter-.2,eligible,margin=0.,enabled=True)
        loss=data+weight*contact;loss.backward();optimizer.step()
        assert parameter.grad.isfinite().all()
        if step%25==0:history.append({'step':step,'measurement_MSE':float(data.detach()),'eligible_proxy_penetration':info['max_proxy_penetration']})
        if step==150:middle={'position':parameter.detach().clone(),'optimizer':copy.deepcopy(optimizer.state_dict())}
    return parameter.detach(),optimizer.state_dict(),history,middle


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--enable-controlled-validation',action='store_true');args=parser.parse_args()
    if not args.enable_controlled_validation:raise SystemExit('Controlled nonpenetration validation is opt-in')
    torch.manual_seed(20261106);times=torch.linspace(0,1,25,dtype=torch.float64)
    truth=.2+.4*(times-.5).abs();observed=truth-torch.where((times-.5).abs()<.1,.04,0.)
    eligible=torch.ones(25,dtype=torch.bool);eligible[12]=False
    # Identical starting vectors/data/Adam settings; the constraint is the only
    # difference. The biased 3D measurement is a manufactured test input, not RGB.
    control,control_opt,ch,_=fit(observed,eligible,0.,300)
    constrained,constrained_opt,ph,middle=fit(observed,eligible,10.,300)
    resumed,_,_,_=fit(observed,eligible,10.,150,middle)
    assert torch.equal(resumed,constrained)
    # The contact mean is over24 eligible points; data mean is over25.
    effective_weight=10*len(observed)/int(eligible.sum())
    expected=torch.where(eligible&(observed<.2),(observed+effective_weight*.2)/(1+effective_weight),observed)
    assert float((constrained-expected).abs().max())<1e-5
    assert abs(float(constrained[12]-observed[12]))<1e-10
    directory=RUNS/'S5_controlled';directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'ground_truth_and_measurements.npz',time_seconds=times.numpy(),
        ground_truth_center_z=truth.numpy(),manufactured_biased_measurement_z=observed.numpy(),
        radius=.2,plane_z=0.,eligible=eligible.numpy(),unknown_index=12)
    torch.save({'control':control,'constrained':constrained,'control_optimizer':control_opt,'constrained_optimizer':constrained_opt,
        'steps':300,'resume150_state':middle,'torch_rng':torch.get_rng_state()},directory/'checkpoint.pt')
    report=REPORT/'S5';report.mkdir(parents=True,exist_ok=True)
    violation=lambda x:float((.2-x[eligible]).clamp_min(0).max())
    write(report/'controlled_validation.json',{'passed':True,'kind':'Manufactured operator/optimizer validation; not RGB/3DGS benchmark or Rubber Duck contact training',
        'known_geometry':'Rigid sphere radius0.2, static planez=0','time_units':'Manufactured seconds, not Rubber Duck acquisition time',
        'biased_measurement_protocol':'Subtract0.04 from center measurements within0.1 seconds of bounce; central observation deliberately ineligible',
        'weight_fixed_before_run':10.,'steps_per_arm':300,'matched_initialization_and_data':True,
        'control_max_eligible_penetration':violation(control),'constrained_max_eligible_penetration':violation(constrained),
        'analytic_optimum_max_abs_error':float((constrained-expected).abs().max()),
        'independent_optimizer_recovery_150_to_300_exact':True,'unknown_observation_not_forced':True,
        'checkpoint_sha256':sha(directory/'checkpoint.pt'),'input_sha256':sha(directory/'ground_truth_and_measurements.npz'),
        'control_history':ch,'constrained_history':ph,'contact_force_friction_restitution':False,
        'does_not_admit_real_scene_contact_surface':True})
    print('CONTROLLED_NONPENETRATION_PASSED',violation(control),violation(constrained),flush=True)


if __name__=='__main__':main()
