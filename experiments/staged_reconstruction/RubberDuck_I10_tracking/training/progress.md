# Progress

- T2 preregistered 775a5fdb; implementation/native/exact-state recovery qualified 2dc32b0a (19 tests).
- Both Gaussian arms completed 2000 updates, with exact matched initial model/two Adam/RNG/camera/physics cache. Static and physics are fully frozen. No color-selected static reference, masks, target replacement or old I10 objective in new training.
- All30 native1600 validation: J6 parent31.994673, RGB31.570412, tracked30.646355dB; tracked-control -0.924057dB fails <=0.3 gate. Training correspondence error0.978803->0.574760px is an optimized fitting diagnostic, not independent 3D validation.
- Independent audit passed: all4000 camera samples, initial full states, immutable static/physics model+Adam/cache, six complete Gaussian PLYs, nine native worldXYZ diagnostics,180 images and all30 metric averages. Both H264 videos independently decode200 frames each.
- Actual33 native/comparison/decomposition/observed/video-contact/3D-path images reviewed. Fragmentation and dark high-alpha tails remain; manual completeness/cleanup failed. Historical color-conditioned parent disclosed; no from-scratch unknown-static claim.
- Final status completed_not_promoted, pending:false. Keep accepted J6 model; no extra training, tuning or paused-job restart.
