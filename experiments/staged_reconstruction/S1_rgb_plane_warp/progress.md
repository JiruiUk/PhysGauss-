# S1-D6 RGB support warp

Status: implemented/tested/frozen, recovery verified; full inference next.
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

Next one serialized two-condition96-view inference, independent192-view parent/
slope/acceptance/hash audit plus fixed0/24/72 full raw-RGB replay, then independent
static GT/structure evaluation. Actual process check found only the unrelated
integrity monitor and no D6 stop/lock. Keep unrelated jobs untouched. No training
promotion from oracle/cost decrease; no paused-task recovery or model change.
