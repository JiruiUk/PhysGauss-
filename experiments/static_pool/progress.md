# R0 progress — reviewed 2026-11-03

Both matched15k→50k arms completed35000 actual updates. Evaluation, exports,
full decoding of16 videos, native visual review and independent evidence audit
are complete. Training core and immutable protocol were unchanged.

H1 is not promoted. Validation image guardrails passed (PSNR -0.017905dB;
LPIPS +4.30%), but fixed-view fragmentation remained. Selected-static validation
blue contribution fell51.60%→12.12%, median opacity0.999996→0.003908, and
training-image track reprojection worsened1.84→7.56px at400px width.
Zero static time variation does not establish correct geometry.

Next step: static selection/reference geometry audit, before R1 or another long
run. No new full candidate or physics module launched. Details and limitations:
`assessment.md`; evidence: `final_audit.json`, `review_decision.json`.
