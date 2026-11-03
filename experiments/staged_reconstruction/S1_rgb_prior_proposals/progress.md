# S1-D4 RGB-validated prior proposals

Status: preregistered, implementation pending. D3/D3-O completed and archived;
do not rerun them. User authorized fixed Small and continuous improvement.
No new Gaussian training, actual depth proposals or claims of artifact cleanup.

Next implement this one protocol behind an opt-in flag. Tests first, then
source/parent/RGB freeze, one serialized diffuse/glossy inference and independent
GT audit. Check processes/locks/stop.request before any launch; preserve current
unrelated local_rigidity training and integrity monitor. No subagents or sweeps.

Do not read S1_pretrained_geometry/oracle_diagnostic outputs as inference input.
Use only frozen raw_relative, D1 depths, D2 strict intervals and RGB sources.
Keep all original candidate membership and unknown identities. No proposal when
bounds/source checks insufficient; no negative background/dynamic label.

Small's thin ordering is imperfect. Its bounds only nominate an alternative;
fit/held RGB acceptance and independent GT audit must establish any gain.
Original source checks are not independent GT. Candidate retention is not
correctness or missing-surface recovery. Strict follow-on evidence and separate
matched native full-state checks are still required before Gaussian training.

## Implementation/preflight complete

22 tests pass (14 new/8 inherited). The native analytic plane fixture initially
showed interpolation bias from fractional source-pixel shifts; preserved
tests_attempt1.log, then used integer shifts at the known analytic plane.
Production cameras/costs/thresholds and the original .03 fixture check unchanged.
Native RGB proposals, held-source rejection/ties, duplicate anchor blocks,
missing/constant/contradictory bounds and an interval with no grid point tested.

Three independent processes compare a four-view analytic inference with a
two-view save followed by two-view continuation. All output arrays and costs
exact, two views restored rather than recomputed. CPU/CUDA disabled calls keep
the identical original tensor. No Gaussian/Adam/RNG training claim. Per-view
atomic output and bound input receipts preserve interrupted/failed files.

Frozen source/parents/RGB/protocol digest:
dac46336a930ad0eec3648ddea8ca847638d353de6ec4aeb67565d3a04dbff77.
Ready for one serialized two-condition proposal inference and independent GT
audit, then actual fixed-image review. No actual candidate outputs yet.
