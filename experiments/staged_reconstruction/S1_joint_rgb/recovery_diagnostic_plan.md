# J1 recovery failure: one read-only diagnostic, no gate change

Full40 versus20+20 failed the original1e-6/1e-5 model tolerance. Initial complete
states and final RNG/camera traces are exact. Training losses already differ
slightly by step4, before the split boundary, so failure cannot yet be attributed
to missing serialization state. Forty-step parameter differences are material
despite similar losses. Formal training remains forbidden; preserve all outputs.

Next read only the saved diffuse/frozen step20. Independently load model and
both Adam payloads into two native scenes and immediately compare every saved
model/optimizer value exactly. Use the next fixed saved camera-stack entry.
Repeat identical-state native RGB+backward three times without optimizer steps;
measure exact RGB equality and per-parameter gradient differences under unchanged
atol1e-6/rtol1e-5. Save aggregate differences, not a new geometry inference.
Also repeat a fixed small original hash-encoder backward at identical input and
embedding values; use the checkpoint's first128 canonical rows and corresponding
time to avoid a new random source. No GT, weights or sampler updates. This only
localizes numerical nondeterminism; it cannot turn the failed full40 gate into
a pass. Any numerical repair must be fixed and checked separately before use.
