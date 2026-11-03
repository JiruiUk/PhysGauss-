# S1--S6 execution continuation

Read `progress.md` and `../staged_reconstruction_plan.md` before using this
runbook. Check psutil/process logs and `runs/pidg_staged_reconstruction/state.json`
first; never duplicate a running trainer. No subagents. Honor `stop.request` and
do not resume user-paused work. Enable CUDA with
`reproduction/scripts/enable_cuda.ps1`; use `envs/pidg-cu128/python.exe`.

S1 and three S2 candidates have completed. The third S2 is conditionally admitted
for sequential reconstruction correction, with **unresolved fixed-view geometry
and physical identity**. Do not represent the admission as geometry success.
Its exact scope is in `S2_geometry_recovery/admission.json` and `assessment.md`.

S3 uses **`variants.staged_reconstruction.state_recovery.train`** with
`--enable-physics-stage`. The original `stage3.train` entry is frozen evidence
of a real-parent smoke rank failure and incomplete graph-cache recovery. The new
entry changes only the independent stage branch: current-world same-bank kNN
and complete recovery. Do not run the old entry or edit frozen training sources.
S3's formal budget is6000 actual updates; its40→80 smoke already passed.

After each formal S3--S6 stage:

1. Check final native val/test and checkpoint status. Run independent
   `experiments/staged_reconstruction/audit_stage.py --stage S3` (or current
   stage), then `review_objects.py --stage S3`. The latter creates native
   observed/group panels and fixed86/fixed188/original200-frame videos.
2. **Actually view** the observed/group panel and native fixed86/fixed188/original
   early/middle/late exports. Write `S3/manual_visual_review.json` with the actual
   checkpoint SHA, `native_images_reviewed: true`, `files_reviewed` mapping
   relative output paths to SHA256, descriptive `findings`, and an honest
   `approve_conditional_continuation` boolean. Retain geometry/identity unknowns.
3. Run `finalize_stage.py --stage S3`. Numeric gates remain dynamic PSNR>=20,
   dynamic-only alpha>=.85 and static PSNR loss from S1<=1dB. Invalid physics
   stencils never count as zero residual. No gate/weight sweeps.
4. Commit the stage's measurements/review separately and update progress/plan.

After S3 review/admission, run
`-m variants.staged_reconstruction.stage4.bind --enable-track-binding` once.
No qualified rows means diagnose rather than invent bindings. Bindings are
immutable and tied to exact S3 checkpoint/topology; late identity is not revived.

For each subsequent stage S4/S5/S6, use
`-m variants.staged_reconstruction.continuation --stage S4 --enable-continuation`.
Before formal training, run `--smoke --stop-after 40`, then
`experiments/staged_reconstruction/verify_smoke.py --stage S4 --capture40`, then
the same smoke entry with `--stop-after 80`, and finally
`verify_smoke.py --stage S4`. Each stage has4000 formal updates and a separate
native evaluator/admission. Do not bypass a failed stage. Full sampler/RNG/Adam
and derived-neighborhood state are preserved. S6 adds static appearance only
and uses0.25 learning rates; static geometry remains exactly S1.

S5 is a **conditional deep-interior prior** on qualified observed rows, not
certified physical contact. Its manufactured sphere/plane optimizer validation
does not certify Rubber Duck surfaces. Actual contact force/friction/restitution,
constitutive inference and unknown forces remain off. If no eligible or active
penetration exists, report this honestly; it is not proof of correct collision.

Keep automation `pidg-s1-s6` active through the whole authorized sequence. Do not
declare the overall task finished at a smoke, audit or single stage. Once all
authorized stage training/reviews and final measured report/plan revision are
complete, report the actual outcome including failures/limitations and pause
this follow-up to prevent duplicate notices. Experimental completion is not a
claim that all geometry or physical-inference goals have succeeded.
