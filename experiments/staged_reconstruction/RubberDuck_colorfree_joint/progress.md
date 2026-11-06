# J7 progress

- User requested color-free static reconstruction images first, then J6 fusion.
- Displayed native c86/c188 control18k source renders before fusion training.
- Fixed3273-point S1-G/control18k source, exact J6 dynamic state, two2000-step
  matched arms and unchanged RGB/self-depth losses. No old static teacher in
  the candidate; inherited dynamic color lineage remains explicitly disclosed.
- Qualification and formal training not started at preregistration.

## Implementation and qualification

- Nine unit tests passed; new behavior is opt-in. The original PIDG and frozen
  J6/source-static code and checkpoints remain unchanged.
- Frozen 505 source/input files; both compositions retain exact J6 non-static
  state and complete dynamic Adam, and exact selected static model/Adam.
- Native composition RGB/alpha/D parity, nonzero intended gradients and
  opacity-only self-depth gradients passed. Initial self-depth is zero for
  each arm's own teacher. Direct-fusion native images were inspected.
- Both arms completed20 qualification updates. Two independent processes per
  arm restored model/two Adam/bindings/RNG/camera/physics graphs exactly;
  native render was exact, gradients met inherited tolerances, and the next
  Adam update using identical saved native gradients was exact.
- First native invocation failed before training because renderer setup had
  not populated the upstream import path. `native_attempt1.log` is retained;
  an external setup/bootstrap invocation passed without changing frozen code.
- Qualification receipt: `v1/recovery.json`. Formal2000-step arms are next;
  no reconstruction quality result is claimed from qualification.

## Formal matched training completed

- Both `old_static_control` and `colorfree_static` completed2000 new native1600
  updates,4000 total. The serial pipeline exited successfully without restart.
- Independent CPU audit passed: exact respective static/Adam sources, exact J6
  non-static/dynamic Adam start, all4000 camera samples and RNG replay, frozen
  static geometry/physics/physics Adam/graph, and full intended Adam counters.
- Complete initial/final model/two-Adam/RNG/camera checkpoints and hash receipts
  are in `runs/pidg_staged_reconstruction/RubberDuck_colorfree_joint/v1/formal`.
- `v1/independent_audit.json` records the full-state checks. Quality evaluation,
  native PNG/PLY/video export and manual review are pending; no promotion.

## Evaluation and actual review completed

- All30 native1600 validation means PSNR: retained J6=31.994673,
  direct replacement=29.393544, old-static control=32.106181,
  color-free joint=34.045075dB. Candidate exceeds control by1.938895dB and
  J6 by2.050402dB; both fixed0.3dB nonregression gates pass.
- Actually inspected35 PNG files: both source static views, all6 fixed
  comparisons/decompositions/candidate native RGBs,6 observed training views,
  selected static RGB/alpha/D,2 summary sheets and2 decoded video contacts.
- Static RGB residuals reduce, but broad static alpha/spikes and dynamic dark
  high-alpha oval/streak support and fragmentation remain. No consistent
  completeness/geometry-cleanup improvement; no geometry promotion.
- Independent final audit passed:505 frozen training inputs/sources,35 eval
  inputs/sources,4000 samples,234 eval PNGs,8 complete Gaussian PLY all fields,
  12 world XYZ diagnostic PLY hashes/parity,30 metric aggregates, and two H264
  faststart videos independently fully decoded200 frames each.
- Complete receipts: `v1/{assessment.md,final_validation.json,manual_review.json,
  final_review.json,final_status.json}`. Status
  `completed_rgb_improved_not_geometry_promoted`, pending:false. Original J6
  remains intact; candidate is retained as an appearance improvement result.
- Training/evaluation processes exited. No extra run, sweep, paused-job resume,
  old-model replacement or physics update is scheduled by this task.
