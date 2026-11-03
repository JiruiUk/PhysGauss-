# S1-G revision2

Authorized next static reconstruction pilot. New branch only; no known static colors/masks, object count, material segmentation or physics losses. Historical PIDG, C1, S1-CT and S3-U sources/weights remain frozen.

- Geometry discovery completed: 3273 common initial points; 19953 verified reservoir voxels; 276523 supported pixels across120 train frames (2.5604% of all pixels; not static recall).
- Actual candidate image review accepted only a bounded pilot; weak-texture interiors and moving-object-edge contamination remain unresolved. Revision1 and its source/output are retained.
- 45 tests passed, including16 new tests and29 prior tests. Opt-in entry points, original PIDG hashes and native renderer parity verified.
- Independent40→80 process recovery passed in both arms. Birth smoke added8 points at30 and60; full Adam/RNG/sampler/confidence/topology replay passed. No old Adam counters/moments reset.
- Formal protocol: geometry and birth arms, same initialization/120-camera pool/order, each12000 steps; progressive confidence every1000. Birth at3000/6000/9000, up to512/event; prune only trial-born points at5000/8000/11000.
- Pending: formal training, checkpoint audit, native comparison against S1-CT, inferred-mask metrics, PLY, fixed86/188 and original-trajectory H264 exports, actual PNG review and plan update. No replacement of S3-U until evidence warrants it.

G1 geometry completed12000 updates; all six Adam counters12000, finite parameters/moments, full RNG/camera/confidence replay and431 frozen source/input/history hashes passed. Points3273; nonzero image supervision12000/12000. Final numerical/native evaluation remains pending G2.

G2 birth completed12000 updates; final3587 points. Formal audit passed all six Adam counters12000, finite parameters/moments, full RNG/camera/confidence/topology and frozen history. Actual topology events: [{"step": 3000, "operation": "birth", "count": 512, "points": 3785}, {"step": 5000, "operation": "prune", "count": 320, "points": 3465}, {"step": 6000, "operation": "birth", "count": 512, "points": 3977}, {"step": 8000, "operation": "prune", "count": 442, "points": 3535}, {"step": 9000, "operation": "birth", "count": 512, "points": 4047}, {"step": 11000, "operation": "prune", "count": 460, "points": 3587}]. Evaluation and actual image review remain pending.
