# D3-O fixed diagnostic before any ordinal/shape result

D3 failed direct static-depth admission. Its high raw rank correlation does not
tell whether anchor calibration or non-affine shape bias dominates. Diagnose
the same frozen192 predictions with a single evaluator-only affine fit, and
fixed raw-image pixel pairs. No new model or weights, no changed D3 selection.

Oracle GT affine fit uses spatially disjoint fit/check pixels. Its coefficients,
depth arrays and masks stay in evaluator memory; only aggregate errors and
colored figures are saved. This is a privileged diagnostic, not deployable
depth, a new Gaussian experiment or proof of a best possible error bound.

Ordinal pairs and strong-prediction condition are defined without GT. GT then
describes static-visible purity, ordering errors on well-separated surfaces,
thin pairs and false ordering on nearly equal depths. Conditioning correctness
on GT static-visible endpoints is an evaluator diagnostic, not a legal
training mask. Empty background and moving endpoints remain separate.

Run only this declared diagnostic. Preserve D3 failure and all old thresholds;
do not export GT-fitted assets into reconstruction. The results will determine
whether to investigate RGB-validated ordinal/proposal evidence, rather than
simply forcing metric/normal priors or choosing a larger pretrained model.
