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
