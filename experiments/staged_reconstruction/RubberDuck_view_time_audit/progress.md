# V0 read-only camera/time diagnostic

2026-11-06: protocol preregistered in 91248fce before new native measurements.
Exact J7 and both J8 continuations; no training or model changes.

- Metadata: 120 exact train times and 120 distinct camera poses; no same-time
  multi-camera training pair. Input metadata and all training RGB are frozen.
- Ten tests passed, including null/undefined denominators, pose accounting,
  deduplicated 482 pairs/model, state hashing and default-off CLI.
- All three models passed exact direct-native RGB/alpha/D parity and gray-render
  alpha/D parity. Full model/two Adam/RNG/camera/physics/controller/flags and
  gradients remain exactly unchanged, including exception restoration.
- Source/input freeze finalized before full measurements. One serial main sweep
  is running; independent full replay and actual visual inspection follow.

No novel-view fixed-time GT exists. Support and dark/high-alpha fractions are
descriptive proxies, not completeness accuracy or object identity.
