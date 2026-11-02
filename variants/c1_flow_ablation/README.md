# C1 repaired: optical-flow on/off

**Current protocol, corrected by the user at16:08 UTC:** do not retrain the
original model. Reuse `runs/pidg_color_repair/C1_repaired/rubber_duck` and train
only O1. The unnecessary new O0 was saved/stopped at9712 and is excluded. Keep
the old queue's stop request; do not run its original two-arm main. Current
entry point is `output/c1_flow_ablation/reuse_control.py`; amended immutable
protocol is `experiments/c1_flow_ablation/reuse_control_plan.json`. Historical
best47k and final50k selections are reused without reselection. New O1 still
starts from the same complete3k state and changes only the flow-supervision
package. Existing C1 training settings and model configuration were checked.

```powershell
# After enabling reproduction/scripts/enable_cuda.ps1, only if no active queue:
envs/pidg-cu128/python.exe -u output/c1_flow_ablation/reuse_control.py --hours 16
envs/pidg-cu128/python.exe output/c1_flow_ablation/status.py
```

The original preparation protocol below is retained for audit; its new-control
retraining instruction has been superseded by the user's correction.

The strongest repaired C-series baseline is C1_repaired by historical validation
PSNR (39.6648946 dB; test40.3721275 descriptive only). O0 and O1 both branch its
unchanged complete pre-flow3k state and train to total50k. Historical final50k
weights are not used as an initialization or substituted for a new control.

Only `observed_flow_enabled` differs. Both retain C1's original momentum
implementation, independent velocity/stress, clock, positive color, mask,
foreground loss and optimizer/densification schedule. No M1 guard, local material
momentum, shared-stress extension or N1 discrete operator is added.

The opt-in loss matches raw train-only UniMatch source-to-target-camera flow
with the actual loader intrinsics, corrected CUDA offsets/compositing weights,
and no reconstruction-derived depth. Half geometric trajectory flow plus half
independent-velocity advected flow, weight.01, every100 steps after3k with2k ramp.
It is one flow-supervision package, not separate identification of its two terms.

Preflight: tests, fresh common20 smoke state, two100-step arms, O1 full70->100
restore, all-flags-off20, isolated actual flow-only optimizer updates and exact
inherited image/time paths. No new parameters or optimizer groups. Two full arms
add47k each; prefix reused, no extra fitting. CUDA nondeterminism is disclosed.

Activate `reproduction/scripts/enable_cuda.ps1`, then run:

```powershell
envs/pidg-cu128/python.exe -u -m variants.c1_flow_ablation.run --preflight-only
envs/pidg-cu128/python.exe -u -m variants.c1_flow_ablation.run --hours 16
envs/pidg-cu128/python.exe output/c1_flow_ablation/status.py
```

The single coordinator uses a file lock and frozen SHA receipts. Both finish
before validation selections are locked and final test opens. Best and final
GT|Render PNGs/videos/ZIP, frozen validation-flow evaluation, source/data/full
checkpoint checks, sample-schedule checks, independent PSNR/pixels/video counts
and a side-by-side gallery are produced. Final manual review remains required.
Do not change running source/config/data or repeat an active queue.
