# Frozen H3 trajectory / velocity-only diagnostic

This is a standalone diagnostic, not a new full-training physics module.
Source: H3 final 50,000-step snapshot. No source model or snapshot is overwritten.
An explicit enabled=false exits without importing model code or touching weights/RNG/CUDA.

## Fixed intervention

- Freeze canonical positions and all deformation/grid parameters. Cache detached full-center derivatives of R(q(X,t))X+b(X,t), h=0.001 normalized time; second-order one-sided endpoints.
- Train a diagnostic copy of the existing PIDG VelocityNet using only normalized MSE to cached derivatives. No momentum, external force, stress, RGB, material or constitutive loss.
- Train its shared features, time embedding, index embedding and velocity head. Stress-head parameters remain frozen, but shared-feature changes can change the unused stress prediction. Never deploy this copy as an unchanged stress model.
- Use only transforms_train.json (120 timestamps) and transforms_test.json in the training loader view (30 validation timestamps). Do not read final-test RGB or metadata.
- Partition opacity>=0.05 eligible particle IDs into80% fitting families and20% held-out families using seed17, then uniformly sample2048/512 Gaussian rows respectively. Both Gaussian rows and inherited particle IDs must be disjoint. Not semantic foreground sampling; report moving-point metrics separately. This corrects the archived spatial-only attempt1, stopped before broad fitting because cloning shares IDs.
- A: 128 fitting particles x 8 training timestamps, full-batch, 4000 steps, Adam 1e-3. Small-set overfit check, not a generalization claim.
- B1/B2/B3: 2048 x all120 training timestamps, 12000 steps, batch4096, pretrained lr1e-3/pretrained lr3e-4/scratch lr1e-3. Same architecture, no capacity increase. Cosine LR decay to 10% initial. Seeds/config fixed before observed fitting results.
- Report fixed final budgets for every arm; do not choose an early checkpoint or arm based on final-test images. Validation is diagnostic across these named arms, not an untouched final test.
- Evaluate fitting pairs, same particles at30 unseen velocity-fitting timestamps, held-out particles at training timestamps, held-out particles at validation timestamps. Embeddings of held-out particle IDs are not updated, so that test measures spatial/index transfer, not merely temporal interpolation.

## Evidence and limits

Check gradient finiteness/nonzero norms, immutable teacher hash/target hash, source-file hashes before/after, unused stress-head parameter hash, repeated-target query equality, finite-difference sensitivity (h=0.00025/0.0005/0.001/0.002), and source diagnostic reproduction.
Relative RMSE is RMS(v-derived)/RMS(derived); zero velocity gives100%. Direction and speed ratio are reported only above a training-derived fixed movement threshold. These are internal reconstructed-trajectory targets, not physical velocity GT.
Good small-set fitting demonstrates an operational gradient path; good broad fitting rules out gross capacity deficiency on these sampled targets. Poor fitting does not prove architecture incapacity. No observation here alone attributes the original error to missing external forces or predicts better RGB/PSNR; the renderer stays unchanged.
