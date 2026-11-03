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

## Two-condition inference, independent GT and actual review completed

All423747/423100 candidate pixels retained. Bounds support211872/210985;
eligible39163/38419; finite proposals38290/37475. Fit nonworse only3/2; final
accepted1/1. Both actual view82 native RGB outputs/costs replay exactly in an
independent process; all192 saved input/bounds/acceptance/membership audited.

Both numeric and strict Gaussian gates fail. Global MAE effectively unchanged
0.036805/0.036826; sole accepted point c26_t3(85,108) worsens GT error
0.009416→0.039775 even though held costs fall. Thin accuracy/coverage unchanged.
No Gaussian/real training or clean-complete claim. Actual fixed0/24/72 RGB/error
figures and sole changed-pixel full/crop figures viewed for both cases.

Next independently audit original photometric costs at privileged GT depths,
separating reference cross-face windows/source visibility/quantization. It is
an evaluator-only measurement audit, not a threshold retry or new pretrained
model. No GT arrays, masks, normals, coefficients or selected points may be
exported as training inputs. Do not rerun D4 or relax old gates.
