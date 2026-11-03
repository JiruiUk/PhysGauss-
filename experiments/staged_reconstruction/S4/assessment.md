# S4 completed measurements

Completed 4000 actual updates; native1600x900 val/test, full-state/sampling audit, native image review and three200-frame video decode checks completed.

Validation full PSNR 28.513441dB, dynamic 25.340218dB, dynamic-only alpha 0.986185, static 36.199356dB. Test dynamic 25.503440dB.

Unchanged numeric gates: {'dynamic_PSNR': True, 'dynamic_only_alpha': True, 'static_preservation_from_S1': True}. Exact Python RNG/camera replay: True. Static geometry exactly matches S1: True. All fixed-frame internal physics stencils valid: True.

Actual visual findings:

- Observed-camera duck silhouettes are recognizable; lower early duck and some contours remain blurred.
- Novel fixed86/fixed188 views still show sheets, speckle and holes; no geometry success established.
- Late model-bank leakage remains and physical identities stay unknown.
- All455 track observations worsen from0.536841 to0.785017px@400; S4 has not demonstrated improved track accuracy. Retain this negative result; no automatic weight increase or promotion as a track improvement.
- Conditional continuation only to the predeclared joint curriculum; S6 uses its already planned lower learning rates. S5 has455 eligible observations but zero active interior violations and cannot demonstrate collision correction.

Manual review scope is the named native observed/group and fixed-view files in manual_visual_review.json. Every exported video frame was independently decoded, but not every video frame was manually watched.

Qualified track mean projection error at400px: 0.7850169922133069. Conditional interior diagnostic: {'eligible_count': 455, 'active_violations': 0, 'max_proxy_penetration': 0.0, 'no_eligible_samples_is_not_collision_validation': False, 'distance_is_conditional_geometry_not_GT': True, 'signed_distance_min': 0.11265810579061508, 'margin': 0.08, 'outside_volume_observations': 0}.

These are training-image correspondences and internal model/conditional-static proxies. They are not independent geometry, material velocity, physical stress or contact GT. Acquisition seconds and late physical identities remain unknown. Novel-view geometry limitations are retained; a successful numeric gate is not geometry success.

Conditional continuation admission: True. Contact force/friction/restitution, constitutive inference and unknown forces remain off.
