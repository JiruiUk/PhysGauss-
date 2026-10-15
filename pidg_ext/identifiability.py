"""Low-dimensional physical-residual Jacobians; conditional, not a recovery claim."""
import torch


def jacobian_diagnostics(residual,theta,material_dimension=0):
    if theta.ndim!=1 or not 0<len(theta)<=128:
        raise ValueError('Identifiability diagnostics require 1-128 parameters')
    J=torch.func.jacrev(residual)(theta).reshape(-1,len(theta)).detach()
    if not torch.isfinite(J).all():raise FloatingPointError('Nonfinite parameter Jacobian')
    singular=torch.linalg.svdvals(J)
    if len(singular)<len(theta):singular=torch.cat((singular,singular.new_zeros(len(theta)-len(singular))))
    maximum=singular[0].item();tolerance=max(1e-10,maximum*1e-6)
    rank=int((singular>tolerance).sum());minimum=singular[-1].item()
    ridge=max(1e-10,maximum**2*1e-6)
    identity=torch.eye(len(theta),device=theta.device,dtype=theta.dtype)
    covariance=torch.linalg.solve(J.T@J+ridge*identity,identity)
    std=covariance.diagonal().clamp_min(1e-300).sqrt()
    correlation=(covariance/(std[:,None]*std[None,:])).clamp(-1,1)
    cross=correlation[:material_dimension,material_dimension:]
    report={'singular_values':singular.tolist(),'min_singular_value':minimum,'max_singular_value':maximum,
            'condition_number':maximum/minimum if rank==len(theta) else 'inf',
            'effective_rank':rank,'parameter_count':len(theta),'residual_count':len(J),
            'rank_tolerance':tolerance,'covariance_ridge':ridge,
            'covariance_proxy':covariance.tolist(),'correlation_proxy':correlation.tolist(),
            'max_material_force_correlation':cross.abs().max().item() if cross.numel() else None,
            'covariance_interpretation':'Unit-noise ridge proxy with other model variables fixed; not calibrated observational uncertainty.'}
    return report,J


def ambiguity_experiment():
    """Manufactured linear balance: mu*u + kappa*v - force*u = observed.

    Report analytic solutions, including a nonunique exact data fit. The prior
    target is deliberately biased, so prior conditioning cannot masquerade as
    observation-only recovery.
    """
    time=torch.arange(32,dtype=torch.float64)*2*torch.pi/32
    u=time.sin();v=time.cos();observed=u+3*v
    cases={}
    cases['force_known'],_=jacobian_diagnostics(lambda theta:theta[0].exp()*u+theta[1].exp()*v-u-observed,torch.tensor([2.,3.],dtype=torch.float64).log(),2)
    cases['stiffness_known'],_=jacobian_diagnostics(lambda theta:2*u+3*v-theta[0]*u-observed,torch.tensor([1.],dtype=torch.float64))
    def balance(theta):return theta[0].exp()*u+theta[1].exp()*v-theta[2]*u-observed
    both=torch.tensor([.5,3.,-.5],dtype=torch.float64);theta=torch.cat((both[:2].log(),both[2:]))
    cases['both_unknown'],_=jacobian_diagnostics(balance,theta,2)
    prior=torch.tensor([1.5,3.,.5],dtype=torch.float64);theta_prior=torch.cat((prior[:2].log(),prior[2:]))
    cases['both_unknown_with_prior'],_=jacobian_diagnostics(lambda t:torch.cat((balance(t),(t[:1].exp()-1.5)*.5**.5)),theta_prior,2)
    for key in cases:
        cases[key]['data_effective_rank']=2 if key.startswith('both') else cases[key]['effective_rank']
    cases['force_known'].update(mu_relative_error=0.,force_relative_error=0.)
    cases['stiffness_known'].update(mu_relative_error=0.,force_relative_error=0.)
    cases['both_unknown'].update(mu_relative_error=.75,force_relative_error=1.5,analytic_solution=both.tolist(),data_residual_norm=balance(theta).norm().item())
    cases['both_unknown_with_prior'].update(mu_relative_error=.25,force_relative_error=.5,analytic_solution=prior.tolist(),data_residual_norm=balance(theta_prior).norm().item(),prior_target_mu=1.5)
    return {'protocol':'Manufactured balance with true mu=2, kappa=3, force=1; analytic solutions, not RGB inversion.',
            'cases':cases,'warning':'The prior raises the objective rank; observation-only rank remains deficient. Convergence is not identifiability.'}
