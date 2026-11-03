# S1-D7 source-plane transport

Status: implementation and independent-process preflight complete; full
192-view inference/evaluation pending. No Gaussian or real training.
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
selection or protocol modification. Full dataset measurements remain unknown.

Next run only protocol.json's frozen cross-view hypothesis mechanism, default
off, with same D6 raw-RGB matcher and source anchor checks: one96-view
inference per case and independent geometry/coverage audit. Check actual
processes/locks/stop.request first; preserve other jobs and paused tasks.
No GT normals/visibility/labels/quantity, no Gaussian promotion from controls
or local means; all unknown identities and original candidate rows retained.
