# Video depth pilot: not qualified as metric geometry supervision

Official Video Depth Anything Small Metric was actually run in an isolated
process. Source4f5ae23172ba60fd7bc11ef671cca678842c7072 and116,444,063-byte
weights SHA2563c28432b4e1f0d7bb31cad5151b6313b49457db5aa58d82e85bfb0f8b1311b33
are pinned. Only training RGB was read; no uploads, PIDG optimizer updates,
renderer changes or shared Python-environment dependency changes occurred.
EasyDict1.13 was placed in a separate workspace vendor directory after the
first inference attempt reported that missing dependency. xFormers was not
installed; upstream's supported fallback was used.

Two fixed configurations were tested: six independent12-frame training
windows, and the entire ordered120-frame training sequence with upstream
keyframes/overlap stitching. Both use short-edge280 inputs, metric weights,
and the same original RGB preprocessing. Missing validation/test frames are
never filled into the sequence. Training timestamps are irregularly spaced;
the model does not receive their true intervals. This is a limited pilot,
not a reproduction or rejection of the complete VDA method.

After one global positive scale fitted to frozen-static reference depth,
the120-frame sequence fails the unchanged six-anchor qualification:

|Training anchor|Held-pixel median error|P90 error|Dynamic extrapolation|
|---|---:|---:|---:|
|2|.879|1.903|5.6%|
|43|1.049|1.711|14.5%|
|86|.205|.546|44.6%|
|146|.997|1.963|.1%|
|188|.581|1.337|13.5%|
|200|.926|2.092|3.7%|

Errors are scene units against frozen model-conditioned static geometry,
not independent depth GT. Required per-anchor median≤.05 and P90≤.15 fail
everywhere. Independent windows also fail. The old pseudo-depth comparison
in the raw records allows separate per-frame inverse-depth affine fits and
therefore is not an equal-calibration accuracy ranking against this global
metric test.

Actual visual review covered pilot anchors43/146 and full-sequence anchor146.
Their common depth color scale exposes large global alignment errors, while
the static error panel has substantial saturated regions. Smooth silhouettes
or visible duck outlines do not establish correct surface depth. No dynamic
geometry or identity correctness has been demonstrated. No depth targets
from this pilot were added to PIDG training.

Two qualification unit tests pass. Raw arrays, exact RGB input receipts,
full outputs, original failed attempt and successful inference logs are
retained. Source-lock's `repository_code_executed:false` records download-time
state; the separate inference receipts explicitly record subsequent actual
execution. Continue the independent I7 trial; do not weaken depth gates.
