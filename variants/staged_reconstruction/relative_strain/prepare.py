"""Preregister I9's single additional prior before any integration outcomes."""
from ..prepare_data import ROOT, REPORT, RUNS, read, write, sha
from ..early_depth.prepare import INITIAL, INITIAL_SHA

EXP = REPORT/'relative_strain'
WORK = RUNS/'relative_strain'


def main():
    assert not (EXP/'protocol.json').exists(), 'Preserve experiment versions'
    assert sha(INITIAL) == INITIAL_SHA
    inherited = read(REPORT/'covariance_transport/protocol.json')
    plan = dict(inherited, arm='I9_relative_strain', relative_strain_enabled=True,
                strain_samples=512, strain_reference_time=1/199, strain_weight=0.02,
                strain_robust_scale=0.3, strain_ramp=0,
                strain_sample_weight='uniform; independent of opacity or visibility',
                strain_times='observed training time and ((global_step-1)%200)/199',
                strain_reference_gradients=True,
                strain_reference_is_known_stress_free=False,
                strain_is_constitutive_stress=False,
                strain_adds_optimizer_parameters=False,
                covariance_transport_objective_retained=True,
                original_physics_and_image_objective_retained=True,
                matched_control='I8 same architecture from original step zero at16k without relative strain; fixed S3-U final6k remains the user-selected final reference',
                original_pidg_control_smoke_updates=100,
                disabled_strain_transport_control_smoke_updates=0)
    WORK.mkdir(parents=True, exist_ok=True)
    write(EXP/'protocol.json', plan)
    print('RELATIVE_STRAIN_PREPARED', flush=True)


if __name__ == '__main__': main()
