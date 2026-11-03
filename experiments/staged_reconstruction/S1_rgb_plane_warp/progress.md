# S1-D6 RGB support warp

Status: complete; limited numeric improvement, both strict gates failed.
No D6 trained Gaussians, new86/188, PLY or video. D5 must not repeat.
See plan.md and protocol.json; inherited gates are exactly the original D4 gates.

29 tests (15 D6 / 14 inherited D4) pass: inverse-depth slope/held/center checks,
slanted world-plane geometry, differentiable raw RGB/depth/slope, default-off
identity and CLI, invalid plane/texture/chroma/source geometry abstention,
and textured integer plane proposal. Initial28-test pass retained; one extra
invalid-current-plane test added, no failing test or adjusted threshold.
Train-only freeze409 files SHA256:
3fbc1cb12072030d011bd9d9223f2923163d8ee3c030a5664f9cd203ba0d0e9c.
No evaluator asset is in the inference input manifest. Three separate processes
compare complete4 against2+2 views for BOTH conditions: every saved array and
file hash match, including coefficients/residuals/costs/acceptance/depth.
Analytic controls measured once under preregistration, without selecting or
adjusting the candidate. No Gaussian optimizer/RNG recovery claim is made.

Two-case192-view inference and independent192-view parent/slope/acceptance/hash
audit plus fixed0/24/72 full raw-RGB replay all pass. Measurementaff05616:
423747/423100 candidates retained;98662/98504 eligible;14731/14655 accepted.
MAE0.03680488→0.03659113 /0.03682616→0.03661146 (about0.58% reduction), gross
errors barely change, thin P90 increases. Numerical5% and Gaussian evidence
gates fail. Updated subset gains cannot establish full geometry/artifact gains.
Actually viewed both fixed-view triptychs and all analytic control curves;
fractional control improves but one slanted pixel minimum worsens. No new
Gaussian/real training/86/188/PLY/video or Gaussian optimizer recovery.

Post-hoc evaluator-only diagnostic (not a changed gate): about98.8% old gross
errors have no admitted target-local slope. Admitted gross points themselves
have approximately59deg unoriented normal error, so slopes are not GT normals.
Next single mechanism: separately preregistered S1-D7 cross-view local-plane
transport from original four fitting sources; retain original source slope
checks, do not relax D6 or re-run it. See next_plan.md and v1/assessment.md.
