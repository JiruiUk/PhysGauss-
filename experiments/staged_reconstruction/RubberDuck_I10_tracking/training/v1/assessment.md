# T2 completed: track fitting improves, reconstruction is not promoted

The requested mask-free trajectory branch was implemented and trained, with 2000 new Gaussian updates in each matched arm. New supervision does not use blue/yellow/semantic/alpha foreground masks, a color-selected static reference, target substitution, foreground/union coverage or parent self-depth loss. Static background and its Adam remain frozen. The exact J6/I10-descendant parent has historical color-conditioned initialization/training; this is not from-scratch unknown-static reconstruction.

T1 train-only image evidence: 4433 full-image texture proposals, 730 bidirectionally consistent tracks, 222 motion candidates, 123 admitted persistent bindings / 116 Gaussian IDs, distributed 98/11/14 by temporal third. These are point IDs, not object count or physical identities. 1432 deduplicated (frame,Gaussian ID) observations cover 96/120 train frames. One fixed 0.01 robust reprojection term supervises the learned 3D positions through the unchanged PIDG deformation field, alongside full original RGB. No topology or physics updates.

| Native1600 all30 validation | J6 parent | RGB only | RGB + 3D tracks |
|---|---:|---:|---:|
| PSNR (dB) | 31.994673 | 31.570412 | 30.646355 |
| SSIM | 0.980425 | 0.979650 | 0.978045 |
| Full-image L1 | 0.003297 | 0.003469 | 0.003882 |
| Training-track mean error (400px units) | 0.874806 | 0.978803 | 0.574760 |
| Training-track P90 error | 1.550111 | 1.698590 | 1.170281 |

The candidate is 0.924057dB below matched RGB control, failing the frozen <=0.3dB nonregression gate. Training correspondences fit about 41.3% better than control, but they are the optimized observations, not independent validation or measured 3D geometry. Continued RGB training itself also drops relative to the parent; the exact single pair does not establish stable causal attribution given the retained native CUDA trajectory sensitivity.

Actual native fixed86/188 early/mid/late RGB and alpha/D decompositions still show fragmented off-trajectory bodies and high-alpha dark tails. c86/t200 retains a bright elongated alpha lobe despite dark RGB, and c188/t100 retains broken bodies and extended support. c188/t200 shows recognizable silhouettes in both arms, without resolving the other camera-time combinations. Six original-trajectory RGB panels look much better than off-trajectory views; index24 still loses part of an observed body behind the current representation, while index72 remains distorted. This fails manual completeness/cleanup. Across the six fixed panels, dark high-alpha support is 126905 control vs 129773 tracked pixels over 8640000 total panel pixels (evaluation-only rendered RGB<0.05/alpha>0.5 proxy, never a training class or GT area). A smaller total alpha support is not a correctness proof.

Static model/Adam and native RGB/alpha/D are exactly unchanged, as are physics model/Adam/cache. Dynamic Adam continues 26k->28k with exact inherited rates/eps1e-15, RNG and camera stack; no reset. 19 tests, native flags-off RGB/alpha/D and gradients, finite nonzero canonical/grid/deform track gradients, 20-step smoke per arm and two independent-process full-state/native-gradient/fixed-gradient-Adam recovery checks per arm pass. Original failed40-step deterministic-trajectory findings remain unchanged.

Full state/sampler/PLY/image/video audit and final review are recorded in the adjacent JSON receipts. Two fixed-view H264 Main/yuv420p/faststart videos include parent/control/tracked, each200 frames; native full Gaussian and world XYZ diagnostics remain distinct. Persistent 3D path plots show model-inferred depth only. Fixed cross-time views have no independent GT; D/alpha is a rendered proxy and black missing depth is not zero error.

Decision: completed_not_promoted. Keep the accepted J6 model. Preserve this actual no-color-reference experiment and its sparse motion-fitting benefit, without claiming artifact removal or correct complete 3D recovery. Sparse persistent-ID reprojection alone did not solve surface/depth ambiguity and uncovered-view reconstruction in this trial. End this fixed candidate; no extra steps, weight/threshold sweep or old paused task restart.
