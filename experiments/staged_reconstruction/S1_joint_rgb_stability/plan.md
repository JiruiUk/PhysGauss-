# J1-R1: fixed common Adam-epsilon numerical control, preflight only

Before implementation/results, fix exactly one numerical control. Preserve
failed J1 full40-vs20+20, its code/protocol/checkpoints and1e-6/1e-5 tolerance.
Immediate full model/two-Adam serialization is exact and identical-state RGB
matches, while native backward differs before any reload. Small gradients with
Adam eps1e-15 may amplify roundoff; this is a hypothesis, not the proven sole
cause. PyTorch Adam defaults to eps1e-8. No epsilon sweep or threshold change.

In a separate opt-in process adapter, use the same frozen J1 training code,
architecture, inputs, initialization, learning rates, loss and6000 parent Adam
moments; change epsilon to1e-8 in both static and dynamic optimizer groups,
identically for frozen/revisable arms. All other optimizer state must remain
exact at construction. This changes the numerical optimizer, so do not call it
unchanged original J1 or original PIDG training. Original code/flags-off and all
historical weights remain untouched; no monkeypatch escapes the worker process.

Preflight ONLY: meaningful switch/no-mutation/epsilon-and-moment preservation
tests, both conditions' native same-value render and finite-gradient3-step
smoke, fresh source/input freeze, then sequential full40 against independent
20+20 for diffuse/frozen first, followed by the remaining three case/arm groups
only if earlier groups pass. Compare complete model/both Adam/RNG/camera/visits
and sample trace at original1e-6/1e-5, discrete fields exact. No repeated trial
to find a lucky passing seed. Failure stops this fixed control; preserve all
states/logs. No retuning, novel masks, GT/depth/normal/DA-V2 or physics inputs.

Even a pass here does not authorize formal6000-step/Rubber Duck training. It
would support a separately recorded matched numerical setup before any new
Gaussian quality measurement. Retain the original four static quality gates
and added structure/dynamic gates unchanged; no gate is relaxed by this audit.
