# V0 complete: read-only camera/time diagnostic

2026-11-06. Protocol 91248fce; implementation/qualification 76ddadcf;
full measurements and independent replay 354056b2.

- Ten tests and all three native/RNG/flags/exception/full-state checks passed.
- 544 input/source files frozen before measurement; exact J7 plus both J8 arms.
- 482 pairs/model, 1446 unique model/pairs, 63 native1600 combinations.
- Independent second-process replay of every800 metric and every800/1600 raw
  render hash passed; 4710 output PNG receipts and full denominator validation.
- Actual visual review: nine RGB crossing sheets, nine alpha crossing sheets,
  nine observation controls, three gray comparisons, two RGB comparisons,
  two numerical curves, six presentation/preview sheets and six native images.
- The view-time-local RGB advantage is supported, especially middle/late, but
  matched-camera gray/alpha also contain large dark support. Not appearance-only.
- Dark fractions are not monotonic structure scores. No simultaneous novel-view
  GT, no formal true generalization error, no unique cause established.
- J8 memory has not removed the pattern. Retain J7; zero training/model changes.

Assessment, manual review and final_review.json close pending review. This
diagnostic is complete; no automatic training, candidate sweep or paused resume.
