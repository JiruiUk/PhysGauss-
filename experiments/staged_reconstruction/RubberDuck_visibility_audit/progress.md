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

- Completed the single serial three-model audit, each all120 native1600 source
  camera/time pairs. Independent process replay: exact metadata, all18 fixed
  targets, source0/24/72 original RGB statistics, all120 aggregate denominators
  and all image hashes match. No state updates in any model.
- Exact metadata census:120 images/120 times/120 camera transforms, zero times
  with more than one camera. J4 joint six dark-support groups have75.88--88.48%
  of source visible mass on dark observations, while11.52--24.12% still falls
  on brighter observations. These are group-level observations, not point
  identity, emptiness/deletion labels or independent surface truth.
- Actually viewed six three-model target comparison sheets, nine source
  visibility sheets, and two additional native1600 RGB/alpha images. Broad
  dark high-alpha support and fragmented off-trajectory RGB remain.
- Final status completed_read_only_not_promoted, pending_review:false. No new
  model/video/PLY/training. Assessment and manual/independent receipts complete.
- Parallel reliable_motion already implements the next time-dependent trusted
  anchor mechanism; reuse that qualification and full completeness evidence
  before a separate static/dynamic joint protocol, without duplicate training.
- Final reproducibility check exposed Git's old autocrlf normalization of
  preregistration/protocol bytes and three qualification logs. Frozen working
  inputs, sources and measurement/artifact hashes all still passed. Retained
  the failed index check, added scoped -text and recorded the existing exact
  working bytes (no protocol/content/threshold changes). The complete source,
  checkpoint, image and task-index byte validation now passes; worker lock is
  released. Both master plans and repository mapping contain only J5 additions
  in the task commit; concurrent worktree edits remain preserved.

## Dependency followup paused after read-only check (2026-11-04)

The parallel reliable_motion graph did not qualify:8592/12882 active parent
points covered (66.6977%), below the frozen80% gate. Its pretraining graph
audit and fixed protocol/array hashes match; this is not a trained-model
quality result. Coverage diagnosis narrows11236 reference-neighbor targets
to10127 after common observations,8643 after the all-common-time radius check
and8592 after rank admission. Expanding currently observed anchors beyond
top2% is exploratory diagnosis only and does not qualify a replacement graph.

The human instructed the dependency task “这轮后结束” (message ID recorded in
dependency_review.json). Its final report was still being prepared when read.
This task does not message, alter, restart or supersede that stopped line and
does not launch dependent joint training. The local pidg automatic followup
is paused; no training/GPU workload belongs to this followup. J5 remains
completed_read_only_not_promoted, not a new failure or new model. The overall
cleanup/completeness goal remains unmet; this does not establish RGB is
unsolvable. Resume only after an explicit human request, then read the actual
dependency final report before a new separately preregistered decision.
