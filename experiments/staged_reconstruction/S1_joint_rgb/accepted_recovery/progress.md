# J1-A progress

Human approved exact state+fixed-native-gradient Adam replay recovery acceptance.
Original40-step/R1 failures retained, all quality gates and original epsilon
unchanged.286-file fresh freeze,2 launcher+original6 tests and all four case-arm
recoveries PASSED: exact model/both Adam/bindings/all RNG/camera; independent
native RGB/alpha exact and raw gradients at original1e-6/1e-5; identical fixture
Adam+constraint update exact. Original40-step failures remain failures.
Next four serial6000-step Gaussian arms with automatic independent evaluation,
PLY/RGB/D-alpha/H264 decode exports, then actual manual review. No cleanup or
Rubber Duck result yet. Source/input freeze and all worker logs verified.

Formal pipeline STARTED after e864d59f recovery milestone. First diffuse/
frozen_static observed training at100/6000; controller/orchestrator live in
runs/pidg_staged_reconstruction/S1_joint_rgb/accepted_recovery. One serial runner
only; do not launch another. Runner proceeds through both arms/two conditions,
independent evaluator and native exports, then leaves manual review pending.
Check real processes,locks,state,logs and stop files before any recovery.
