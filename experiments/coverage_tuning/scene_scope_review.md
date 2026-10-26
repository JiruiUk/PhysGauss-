# Correction: Rubber Duck contains multiple objects

During the user-requested C1/C2 GT/render review, original frame1 clearly shows
multiple distinct ducks and blue obstacles. This contradicts the earlier
single-object description and the assumption encoded as `single_object=True`.

The actual C2/C3 affine loss selects Euclidean canonical neighbors within a
radius. It has no real-scene object-ID bank. A unit-tested helper can gate edges
when explicit IDs are supplied, but those IDs are not supplied in these runs.
Therefore neither cross-object coupling nor contact sticking is ruled out.

The completed runs remain valid observations of this *unsegmented spatial
affine regularizer*. They must not be reported as object-isolated regularization.
C1 mask supervision is a union foreground alpha objective and does not assume a
single object. C3 retains its predeclared settings for a comparable cumulative
ablation; the priority render does not modify a model, loss, or checkpoint choice.

An object-isolated follow-up needs verified known object IDs propagated to
Gaussian children, followed by fresh controlled training. It cannot be obtained
by relabeling these results or editing the assumption flag after training.

Evidence: `runs/pidg_coverage_tuning/C1_mask/rubber_duck/test/ours_50000/gt/00000.png`
(original acquisition frame1). This inspection occurs after the explicit
user-authorized early test export, recorded in `priority_render.json`.
