# S1-D7 source-plane transport

Status: complete_rejected_geometry_gate. Full192-view inference, source/RGB
replay, independent GT/aggregate/data-hash audits and manual review complete.
No Gaussian or real training. Do not repeat this completed candidate.
D6 complete, limited local improvement but failed both gates; do not repeat.

40 tests passed (11 D7 plus inherited D6/D4 checks), including geometry,
coordinate transport, parallel/behind-camera/footprint rejection, gradients,
default-off object identity and stable acceptance/tie behavior. The initial
39-test pass is retained separately; no failed test or parameter change.
Train-only inputs and sources frozen: 809 files, receipt SHA256
84d3730388d9cbe319b037527bdb9f926eea0fa0b6b34cd2a57629ef522ae29a.
Three independent processes completed full4 versus split2+2 for both cases:
all arrays and eight NPZ hashes match exactly, including source queries,
planes, costs, acceptance and unchanged parent depths. This is inference
recovery, not Gaussian Adam/RNG/camera recovery.

Fixed analytic A/B/C controls each accept both supplied-plane hypotheses and
recover their analytic depths from target4.8. Their source planes are fixture
inputs, not evidence that dataset source planes are correct. No control-based
selection or protocol modification. They do not qualify dataset source planes.

Implementation/recovery committed9b2d4ed4; complete measurementseca285b4.
All423747/423100 candidates retained;58279/58201 updates, including25119/25226
without admitted target-local slope. Static MAE0.03659113 to0.03590450 and
0.03661146 to0.03591997, about1.88/1.89% improvement. Thin errors and coverage
improve, but gross/no-static stays20.13/20.15%; both original gates fail.
Only4391/85908 and4443/85836 parent gross/unknown rows have a usable source
hypothesis. Independent posthoc aggregates only, no GT feedback/point assets.
Both fixed0/24/72 RGB/error panels and all analytic controls actually viewed.
See v1/assessment.md and final_review.json. No native86/188/PLY/video claims.

Next is one preregistered full-RGB static-revocation feasibility audit J0,
following the user's joint-cleaning question. Reuse, do not duplicate, parallel
static_dynamic_coupling audit9a9c8fb6. No joint or depth-supervised training is
admitted. Preserve other jobs/paused work, legacy lineage and unknown identity.
Details: next_plan.md and ../S1_joint_revocation_audit/protocol.json.
