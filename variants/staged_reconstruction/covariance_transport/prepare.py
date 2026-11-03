"""Preregister I8 before native integration and formal optimization."""
from ..prepare_data import ROOT, REPORT, RUNS, read, write, sha
from ..early_depth.prepare import INITIAL, INITIAL_SHA

EXP = REPORT/'covariance_transport'
WORK = RUNS/'covariance_transport'


def main():
    assert not (EXP/'protocol.json').exists(), 'Preserve experiment versions'
    assert sha(INITIAL) == INITIAL_SHA
    inherited = read(REPORT/'invertible_motion/protocol.json')
    plan = dict(inherited, arm='I8_covariance_transport', covariance_transport_enabled=True,
                original_covariance_heads_retained=True, original_covariance_heads_inactive_when_enabled=True,
                original_grid_and_deformation_networks_inactive_when_enabled=True,
                reference_covariance='Learnable bounded canonical scale, unit canonical quaternion; no time-dependent covariance head',
                covariance_transport='Dphi(X,t) Sigma_canonical(X) Dphi(X,t)^T',
                canonical_reference_is_known_stress_free=False,
                analytic_chain_derivative_keeps_coordinate_parameter_time_gradients=True,
                native_renderer_covariance_input=True, renderer_source_modified=False,
                separate_covariance_transport_optimizer_group=False,
                extra_depth_or_material_or_force_unknowns=False,
                initial_native_max_abs_tolerance=3e-4,
                initial_native_mean_abs_tolerance=1e-6,
                matched_control='Original S3-U from original step zero and fixed final6k reference; I7 from same original initialization provides a position-map-only16k ablation but failed its original warmup admission',
                no_relaxed_gate_for_failed_historical_covariance_parity=True)
    WORK.mkdir(parents=True, exist_ok=True)
    write(EXP/'protocol.json', plan)
    print('COVARIANCE_TRANSPORT_PREPARED', flush=True)


if __name__ == '__main__':
    main()
