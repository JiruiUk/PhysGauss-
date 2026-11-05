# Next: temporal support before local motion training

T3/T4 are complete, not admitted, with zero optimizer steps. No automatic extension of these fixed trials. Preserve all frozen code, parent weights, original gates and unknown identities.

The observed bottleneck is temporal/spatial evidence: T4 starts from177/25/20 RGB motion candidates and leaves104/7/5 qualified local tracks. A correct short stationary-world explanation is not a permanent static identity, and the unified dynamic pool may contain regions that stop moving. These are possible causes to test, not measured facts about every rejected track.

A distinct next protocol can test cross-window continuation among all original730 RGB-consistent tracks. Admit a stationary-compatible or unknown short-window observation only through independently verified overlap with persistent motion-supported observations, reciprocal model-conditioned binding and native visibility; retain competing/occluded associations as unknown. Do not assign ownership by color, proximity alone, object count or GT, and do not count untested links in the present results.

Only after fixed reliable temporal/neighborhood admission passes, preregister one soft local motion/shape prior and a matched Gaussian test. Preserve local deformation freedom for soft ducks; no global rigid-body assumption. A parent-derived local shape can preserve wrong geometry, so its agreement is not a completeness metric. Native flags-off/gradient checks, the user-approved exact twoAdam/RNG/camera recovery, full matched control, all30 RGB nonregression and fixed-view RGB/alpha/D manual completeness/tail cleanup remain mandatory. No training from failed T4, no threshold/weight sweep, no old paused-job restart.
