# S3 completed measurements

Completed 6000 actual updates; native1600x900 val/test, full-state/sampling audit, native image review and three200-frame video decode checks completed.

Validation full PSNR 28.466855dB, dynamic 25.392299dB, dynamic-only alpha 0.985505, static 35.941669dB. Test dynamic 25.557255dB.

Unchanged numeric gates: {'dynamic_PSNR': True, 'dynamic_only_alpha': True, 'static_preservation_from_S1': True}. Exact Python RNG/camera replay: True. Static geometry exactly matches S1: True. All fixed-frame internal physics stencils valid: True.

Actual visual findings:

- Observed-camera ducks remain recognizable, with blurred edges and lower early duck distortion.
- Fixed86 early/late and fixed188 early/middle still have strong sheets, speckle and holes; no complete novel-view geometry improvement established.
- Early groups are distinct, but frame42 mixture and later green/purple bank leakage remain; late physical identity is unknown.
- Static shape remains stable; permit only the predeclared conditional reconstruction correction, not contact-surface or stress/physical-ID inference.

Manual review scope is the named native observed/group and fixed-view files in manual_visual_review.json. Every exported video frame was independently decoded, but not every video frame was manually watched.

Qualified track mean projection error at400px: None. Conditional interior diagnostic: None.

These are training-image correspondences and internal model/conditional-static proxies. They are not independent geometry, material velocity, physical stress or contact GT. Acquisition seconds and late physical identities remain unknown. Novel-view geometry limitations are retained; a successful numeric gate is not geometry success.

Conditional continuation admission: True. Contact force/friction/restitution, constitutive inference and unknown forces remain off.
