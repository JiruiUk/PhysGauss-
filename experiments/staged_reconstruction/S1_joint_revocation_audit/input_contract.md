# Original RGBA-to-RGB target contract, before new audit code/results

The original Rubber Duck PNGs are RGBA; RGB behind transparent pixels is not
black. Reading them as straight RGB would silently change the old black-
background target. J0 uses the original `stage2.train.image` resize followed by
`RGB * alpha`, exactly the inherited native training target. Alpha is used only
to flatten the stored image against black, not as a static/dynamic label, loss
mask, opacity target, support selection or new segmentation supervision.

The fixed original indices map to frame numbers2/42/82/122/162/200. No frame
selection is based on loss, gradients or evaluator geometry. The original RGBA
files, metadata, preprocessing implementation and legacy model lineage must be
bound by the audit source/input freeze. No training or geometric result has
been measured for J0. This clarifies the preregistered original full-RGB target
contract without changing candidates, frames, tolerances or the objective.
