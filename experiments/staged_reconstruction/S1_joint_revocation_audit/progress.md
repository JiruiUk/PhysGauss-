# J0 progress

Complete: implementation/preflight b7e304ba, six-frame native measurement and
independent review c9ebd762. No training or optimizer step.
9 context/flags-off/RNG/failure/value-restoration/gradient/CLI tests passed.
391 inherited input/source files frozen, receipt SHA256
98bfc2928320238682486f2c08909dea9acd68e212f60e0c72114603bed1ccc0.
Two independent pinned-checkpoint reloads produce exact800/1600 native RGB and
alpha and exact parameter hashes; all six gradient groups finite/nonzero and
agree within originalatol1e-6/rtol1e-5. Caller flags/values/RNG preserved.

Two failed starts retained: missing inherited Ninja/MSVC PATH was fixed by
existing enable_cuda.ps1; the source receipt then encountered Torch's virtual
_ops.py. Independent receipt enumeration now skips nonexistent virtual paths,
with2 tests and separate repair freeze. Frozen measurement code and original
protocol/tolerances/weights unchanged; failure logs/raw arrays preserved.
RGBA alpha is only the original black-background RGB flattening contract,
never a new static/dynamic loss mask; see input_contract.md.

Six-frame audit,54 old images, independent checkpoint/source/gradient/denominator
checks and manual native86/188 review completed. All six gradient groups finite
and nonzero; values/flags/leaf gradients/RNG unchanged. Gradient path qualified,
cleanup/geometry NOT proven. See v1/assessment.md and final_review.json.
Do not repeat the completed measurement.
Check actual processes/locks/stop files and newer parallel coupling work first;
reuse existing9a9c8fb6 audit, do not duplicate it or change other jobs.
D7 is completed/rejected; failed depths/normals do not enter this audit.
No new-label-free-training claim: inherited S3-U used historical conditional
color masks. New measurements use only full train RGB and preserve unknowns.
