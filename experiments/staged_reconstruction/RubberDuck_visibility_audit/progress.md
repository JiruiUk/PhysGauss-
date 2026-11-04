# Progress

- J4 retained as completed_not_promoted; no repeated training.
- Reused static_dynamic_coupling and motion-integrity/I5/I6/I7 negative results.
- One read-only J5 question and protocol fixed before measurements. No training
  or inferred physical static/dynamic labels are produced by this audit.

- Reused uncertainty_motion's completed native120 color-contribution audit:
  first20 buffers are capacity limited and post-T diagnostic weights are not
  compositing weights. J5 tests off-trajectory dark subsets and J4 checkpoints;
  it does not repeat or replace the 2195 RGB-supported anchor measurement.
- Five original tests plus the independent pixel-center calibration test pass.
  Original qualification0 failed only in the evaluation camera fixture; its
  frozen source/log are retained. Separate qualification_repair.py corrects
  the principal point to ((w-1)/2,(h-1)/2), without changing the protocol.
- 209 original files remain frozen; repair has a supplemental freeze. Two
  corrected independent processes (2/3) agree exactly on target group hashes,
  source0 statistics and fixture. Packed/unpacked visibility and native
  RGB/alpha/D parity pass; full model/twoAdam/RNG/camera/graph state unchanged.
  Native fixture: stored weight equals alpha*(1-alpha) exactly, versus actual
  compositing alpha maximum error 0.569745. Stored weights are never used.
- Formal measurement remains zero optimizer updates. Separate metadata/source
  replay is required before final assessment; no training promotion follows.
