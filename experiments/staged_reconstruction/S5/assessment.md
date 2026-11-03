# S5 completed measurements

Completed 4000 actual updates; native1600x900 val/test, full-state/sampling audit, native image review and three200-frame video decode checks completed.

Validation full PSNR 28.857179dB, dynamic 25.603743dB, dynamic-only alpha 0.984816, static 35.983147dB. Test dynamic 25.745418dB.

Unchanged numeric gates: {'dynamic_PSNR': True, 'dynamic_only_alpha': True, 'static_preservation_from_S1': True}. Exact Python RNG/camera replay: True. Static geometry exactly matches S1: True. All fixed-frame internal physics stencils valid: True.

Actual visual findings:

- Observed-camera duck silhouettes are recognizable, with blurred contours and early lower-duck streaking.
- Novel fixed86/fixed188 views retain fragmented sheets, speckle and holes, especially at times different from the fixed camera observation; no geometry success established.
- Late model-bank leakage remains; physical identities after ambiguous occlusion stay unknown.
- All455 qualified track observations average0.66467054px@400, worse than the S3 baseline0.53684143; this is not a demonstrated track improvement.
- All800 conditional interior calls during S5 training have zero active violations and zero nonzero prior calls. There was no actual collision correction; the controlled operator fixture is not real-scene contact validation.
- Conditional continuation is limited to the predeclared S6 reconstruction curriculum with lower learning rates and static appearance updates. Original-plan object-object nonpenetration, contact force, friction and restitution remain unvalidated/off.

Manual review scope is the named native observed/group and fixed-view files in manual_visual_review.json. Every exported video frame was independently decoded, but not every video frame was manually watched.

Qualified track mean projection error at400px: 0.6646705402450247. Conditional interior diagnostic: {'eligible_count': 455, 'active_violations': 0, 'max_proxy_penetration': 0.0, 'no_eligible_samples_is_not_collision_validation': False, 'distance_is_conditional_geometry_not_GT': True, 'signed_distance_min': 0.10471511632204056, 'margin': 0.08, 'outside_volume_observations': 0}.

Whole-run conditional-interior activation: {'objective_calls': 800, 'eligible_observation_instances': 16000, 'active_violation_instances': 0, 'calls_with_any_active_violation': 0, 'maximum_proxy_penetration_during_training': 0.0, 'nonzero_prior_objective_calls': 0, 'all_recorded_prior_values_finite': True, 'not_collision_ground_truth': True}.

These are training-image correspondences and internal model/conditional-static proxies. They are not independent geometry, material velocity, physical stress or contact GT. Acquisition seconds and late physical identities remain unknown. Novel-view geometry limitations are retained; a successful numeric gate is not geometry success.

Conditional continuation admission: True. Contact force/friction/restitution, constitutive inference and unknown forces remain off.
