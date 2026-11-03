# Depth evidence follow-up

S1-N fully evaluated/rejected at1e5af70a; D0 read-only depth evidence/quantization
audit and the fixed next protocol committed atbe408aa4. No real nuisance
continuation is authorized. See assessment.md and refinement_protocol.json.

D1 implementation: default-disabled three-knot inverse-depth quadratic proposal,
half-cell bound, unchanged four-source fit plus disjoint two-source nonworsening
validation. Invalid, nonconvex, zero-step or worsening updates retain the
original depth. No additions/pruning, no GT selection, no Gaussian optimization.
18 tests pass (8 new/10 inherited), including exact disabled object identity
and one worsening held source refusal. Sources/raw-depth inputs/protocol frozen
at53292fe1e0731b9b072c52d104ce7da44e9f91c0480d2dfcc65500151d3924cf.

Next run one serialized96-view-per-case refinement/evaluation. Per-view outputs
allow exact preservation of incomplete work; user stop forbids automatic resume.
Check runs/pidg_staged_reconstruction/S1_depth_refinement/v1/{state,pipeline}.json,
actual processes, both lock files and stop.request before starting anything.
Inference does not load evaluator labels; the separate runner verifies all
dataset hashes before independent GT evaluation. Camera membership and the
original normal-evaluation pixel set stay fixed. No promotion from average
error alone; this is not complete Gaussian training or static-GT cleanup.
