# S6 completed measurements

Completed 4000 actual updates; native1600x900 val/test, full-state/sampling audit, native image review and three200-frame video decode checks completed.

Validation full PSNR 30.585972dB, dynamic 28.327031dB, dynamic-only alpha 0.985981, static 36.169914dB. Test dynamic 28.563049dB.

Unchanged numeric gates: {'dynamic_PSNR': True, 'dynamic_only_alpha': True, 'static_preservation_from_S1': True}. Exact Python RNG/camera replay: True. Static geometry exactly matches S1: True. All fixed-frame internal physics stencils valid: True.

Actual visual findings:

- Observed-camera silhouettes and contours appear sharper than S5, including the early lower duck and later visible ducks; conditional static support remains visually recognizable.
- Novel fixed86/fixed188 views at other times still have fragmented sheets, holes, speckle and stretched geometry. Complete multi-view geometry has not been recovered.
- Model-bank leakage remains, including a long late green streak. Bank colors are model contributions, not independent physical identity evidence; late physical IDs stay unknown.
- All455 qualified training-image track observations average0.42710676px@400 versus S3 baseline0.53684143, an approximately20.44percent decrease. The S4 intermediate worsening remains preserved. This result applies only to the fixed qualified observations, not independent 3D/material-point or occlusion re-identification truth.
- All800 conditional-interior calls and16000 eligible observation instances in S6 had zero active violations and zero nonzero prior calls. S5 and S6 do not demonstrate actual collision correction; original object-object/contact-force/friction/restitution goals remain unvalidated/off.
- Accept this predeclared final checkpoint for conditional reconstruction reporting only. No additional stage, weight sweep, force or constitutive experiment is started by this review; original full geometry/physical/contact scope remains incomplete.

Manual review scope is the named native observed/group and fixed-view files in manual_visual_review.json. Every exported video frame was independently decoded, but not every video frame was manually watched.

Qualified track mean projection error at400px: 0.42710675626461. Conditional interior diagnostic: {'eligible_count': 455, 'active_violations': 0, 'max_proxy_penetration': 0.0, 'no_eligible_samples_is_not_collision_validation': False, 'distance_is_conditional_geometry_not_GT': True, 'signed_distance_min': 0.11077552288770676, 'margin': 0.08, 'outside_volume_observations': 0}.

Whole-run conditional-interior activation: {'objective_calls': 800, 'eligible_observation_instances': 16000, 'active_violation_instances': 0, 'calls_with_any_active_violation': 0, 'maximum_proxy_penetration_during_training': 0.0, 'nonzero_prior_objective_calls': 0, 'all_recorded_prior_values_finite': True, 'not_collision_ground_truth': True}.

These are training-image correspondences and internal model/conditional-static proxies. They are not independent geometry, material velocity, physical stress or contact GT. Acquisition seconds and late physical identities remain unknown. Novel-view geometry limitations are retained; a successful numeric gate is not geometry success.

Conditional continuation admission: True. Contact force/friction/restitution, constitutive inference and unknown forces remain off.
