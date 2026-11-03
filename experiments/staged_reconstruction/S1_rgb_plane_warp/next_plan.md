# After D6: one source of geometry hypotheses from other views

D6 improves mostly small existing errors. Its target-local D1 plane requirement
excludes about98.8% of original gross errors; the few admitted gross points
have poor inferred normals. Do not relax that check or call normals trusted.
Keep D6's saved limited result, with no Gaussian training promotion.

Next preregister S1-D7 cross-view local-plane transport. Use exactly original
four fitting source views. At the projection of each current D6 target point,
query the nearest source pixel; only an original D1 source plane that passed
the unchanged D6 neighbor-fit/held/center checks may supply a hypothesis.
Intersect that source plane with the reference ray, transport its inverse-depth
slope, and require the intersection to project inside the source's5x5 footprint.
Target-local slope admission is not needed to receive such an external-view
hypothesis. The source's original admission test is retained without adjustment.

These are at most four geometric hypotheses, unrelated to object/instance
count. Use the same D6 raw-RGB warp for the current incumbent and each valid
transported hypothesis, same best3/4 fit and two held nonworse/strict-improvement
equation. No GT plane/visibility, normal optimization, new model, new loss,
threshold sweep or point growth. Bad source planes/occlusion/repeated texture
can still cause wrong updates; fixed independent geometry gates must decide.

Full D7 details and inherited gates are declared before code/results in
../S1_plane_transport/plan.md and protocol.json. It is a new inference mechanism,
not acceptance of D6 for training. Any later Gaussian stage still needs the
separate paired protocol and complete native recovery/GT structure audit.
