"""Record qualification and freeze separate evaluation adapters before outputs."""
from pathlib import Path
from variants.staged_reconstruction.i10_tracking.io import ROOT, read, write, sha

B=ROOT/'experiments/staged_reconstruction/RubberDuck_surface_support'
V=B/'v1'
assert read(V/'recovery.json')['passed']
assert read(V/'native_check.json')['passed']
assert read(V/'unit_tests.json')['passed']
assert read(V/'initial_coverage.json')['passed']
assert not (V/'evaluation_freeze.json').exists()
files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [
    B/'controller_audit.py',B/'structural_diagnostic.py',B/'register.py',Path(__file__)]}
write(V/'evaluation_freeze.json',dict(files=files,created_before_formal_training=True))
with (B/'progress.md').open('a',encoding='utf-8',newline='\n') as f:
    f.write('''
- Qualification completed: 27 relevant tests, native1600 exact flags-off RGB/alpha/D,
  original gradient tolerance, finite intended canonical/grid/deformation/scale
  gradients, exception RNG/flag restoration and no mutation. Both20-step smoke
  runs and four independent recovery workers passed model/twoAdam/bindings/RNG/
  camera/physics/controller exact restoration and same-native-gradient Adam replay.
- Initial60 local patches /337 patch-local edge applications /275 Gaussian rows;
  temporal thirds59/1/0. This is a local early-biased pilot, not global coverage.
- Preparation attempt1 stopped before updates/source freeze on a missing individual
  footprint. Attempt2 preserves missing footprints as unknown; no gates changed.
  Both attempts and test logs retained. Separate evaluation adapters frozen before
  formal outputs. Formal matched2000+2000 training is next, not yet performed.
''')
print('L5_QUALIFICATION_RECORDED')
