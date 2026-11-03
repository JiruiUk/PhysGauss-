# S1-D6 RGB support warp

Status: preregistered before implementation/results. No D6 inferred depths,
trained Gaussians, new86/188, PLY or video. D5 is complete and must not repeat.
See plan.md and protocol.json; inherited gates are exactly the original D4 gates.

Next implement the one isolated default-off RGB matching candidate and tests,
freeze original D1/train inputs and sources, verify independent-process complete
inference recovery, then run one serialized two-condition96-view inference and
independent static GT/structure audit. No training promotion from an oracle or
a cost decrease. Do not recover paused tasks, change Small, add GT support or
stop unrelated jobs. Check actual processes, locks and stop.request first.
