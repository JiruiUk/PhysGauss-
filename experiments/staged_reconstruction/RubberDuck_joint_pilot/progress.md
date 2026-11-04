# Progress

- J3 complete, numerical gates passed but geometry cleanup not promoted; retained.
- User requested a real Rubber Duck joint experiment. Parent static weights were
  verified exactly equal to S1 final12k; its full step12000 Adam can be inherited.
- J4 is independently preregistered. No new training result exists yet.

- Implementation 3f1d8974: 10 tests passed; 198 train-input/source files frozen.
  Native1600 flags-off scene identity, exact initial RGB/alpha/D, RGB gradient
  tolerance, isolated opacity-only depth gradient passed. Preparation comparator
  CPU/CUDA mismatch happened before any updates/freeze and is retained separately.
- Both arms completed20 qualification updates, each followed by two independent
  restores. Full model/twoAdam/bindings/RNG/camera/graphs and fixed-native-gradient
  next Adam update are exact. Original native full40 failures remain unchanged.
- Formal paired2000 updates are now starting; no quality conclusion yet.
