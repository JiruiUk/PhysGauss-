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

## D1 completed and reviewed

Both96-view cases completed; all input/output/source/data hashes and candidate
membership passed. Original→refined static-depth MAE0.03800→0.03680 and
0.03802→0.03683; same-pixel normal means30.842→26.744deg and30.866→26.775deg.
Thin errors/P90 improve modestly. About20.3% still disagree with static GT by
more than0.05 or lie on no static surface. Actual c00_t1/c08_t0/c24_t0 depth/
error/normal panels for both cases reviewed; missing evidence and noisy edges
remain. This is a depth-evidence result, not Gaussian artifact cleanup.

See refinement_v1/assessment.md and final_review.json. Do not restart D1 or
change its fixed parameters; retain local numerical improvement only. Next
single mechanism is RGB-only source-view uncertainty/interval evidence,
predeclared before outputs and independently audited before a new paired
Gaussian synthetic training protocol. No strong blanket depth/normal loss,
S1-N real continuation, S3-U replacement or paused task resumption follows.
