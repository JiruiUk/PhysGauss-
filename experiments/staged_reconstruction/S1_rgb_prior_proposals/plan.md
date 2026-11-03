# S1-D4: one RGB-validated relative-order depth proposal

D3/D3-O show that fixed Small has coarse relative order but unsafe metric shape,
thin order and background behavior. Do not use it as static labels or a loss.
This one candidate asks whether it can suggest a different RGB depth basin,
whose acceptance is controlled by the original calibrated RGB sources.

Inputs: frozen D1 candidate pixels/depths, D2 tight anchor depths/intervals,
Small raw relative outputs and original six train-source indices/RGB/cameras.
Never read D3 calibrated depths or any oracle diagnostic/GT-fitted asset. Keep
all D1 candidates and depths by default. No growth, deletion, masks, object
counts, physical identities, model finetuning or new Gaussian loss.

At an existing D1 pixel, compare raw relative depth to spatially distributed
strict D2 anchors. Only strong rank differences form one-sided depth bounds;
use robust fixed quantiles of anchor intervals, not an affine prior fit. Fewer
than four distinct8px anchor blocks on a side supplies no bound on that side.
Missing or contradictory bounds mean unknown/no proposal. All old candidates
remain recoverable and are never classified dynamic/background.

Only if the old depth falls outside these bounds, search the unchanged192
camera-derived inverse-depth grid within the bounds. The original four RGB
sources propose by their best-three average photometric cost. Accept a finite
new proposal only if that fit cost does not worsen and both original held
source costs do not worsen, with at least one strictly improving. No rank loss,
GT or appearance mask accepts a proposal. Original sources are acceptance
evidence, not independent evaluation; the latter uses frozen synthetic GT.

This changes which basin can be proposed relative to D1's local refinement;
it is a separate declared candidate, not a retry/threshold adjustment of D3.
Test analytic ordering, bounds/empty/contradictory cases, source validation and
exact disabled identity. Freeze code/data before one serialized two-condition
run, retain all per-view/failed files, inspect fixed figures and audit metrics.
No automatic model/weight/threshold sweep or additional queued candidate.

Preserving membership avoids hiding errors by deletion; it cannot create
coverage in missing D1 rays, which must remain a stated limitation. First
measure correction/error tails and thin accuracy. Mean improvement alone does
not grant Gaussian admission. The protocol's strict follow-on evidence gate
and a separately matched native/Adam/RNG/camera protocol remain mandatory.
