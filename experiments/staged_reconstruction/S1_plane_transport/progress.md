# S1-D7 source-plane transport

Status: preregistration only, before code/results. No inference or training.
D6 complete, limited local improvement but failed both gates; do not repeat.

Next implement only protocol.json's cross-view hypothesis mechanism, default
off, with same D6 raw-RGB matcher and source anchor checks. Test and freeze,
verify independent complete4/2+2 recovery for both conditions, then one96-view
inference per case and independent geometry/coverage audit. Check actual
processes/locks/stop.request first; preserve other jobs and paused tasks.
No GT normals/visibility/labels/quantity, no Gaussian promotion from controls
or local means; all unknown identities and original candidate rows retained.
