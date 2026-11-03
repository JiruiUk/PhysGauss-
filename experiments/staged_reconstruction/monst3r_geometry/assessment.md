# Adjacent MonST3R pairs fail geometry qualification

The official pinned model actually ran all12 preregistered pairs in a separate
process using float32 tensors and official pair-forward mathematics. CroCo's
official TF32 setting and uncompiled PyTorch RoPE fallback apply. Strict weight
loading succeeded after verifying/restoring eight deduplicated tied aliases.
Peak process CUDA allocation is approximately2.65GB, not total system VRAM.
The full MonST3R global/flow/SAM pipeline was not run.

At fixed confidence>=3, eroded dynamic foreground support is:

|Anchor|Two partner IDs|Dynamic confident fraction|
|---|---|---|
|2|3 /6|0% /0%|
|43|42 /40|0% /0%|
|86|87 /88|22.42% /0%|
|146|147 /148|41.84% /33.36%|
|188|187 /186|100% /100%|
|200|199 /196|100% /100%|

The original pilot requires>=75% at each pair. Early/middle static confidence
support is also insufficient: anchors2/43/86 have at most0–48 qualifying static
pixels before further reference filters, below1000. The first qualification
stopped rather than fitting scale with insufficient support. A reporting-only
repair subsequently records unsupported pairs as failures and continues to
measure supported ones; it changes no thresholds or predictions.

High confidence in late frames is insufficient. At188, held-out static depth
error medians are0.243–0.262 scene units andp90 errors0.539–0.599; at200 they are
0.290–0.296 and0.634–0.754. Limits remain0.05/0.15. The frozen static reference
is model-conditioned, not independent geometry truth.

Late dynamic pointmaps also disagree with the known camera: raw point-to-pixel
projection error medians are27.50–29.99 pixels at188 and32.61–33.72 at200;
p90 errors are53.04–57.77 and65.65–67.69 pixels at512x288. The paired changes
in late scaled depth do not fix those failures. Confidence is not a probability
or a certificate of geometry. All12 pairs fail at least one preregistered check.

`pair_qualification.json` retains every measured pair and unavailable metric;
`confidence_audit.json` independently checks necessary support conditions for
all12. The six-row RGB/confidence/support sheet was actually inspected:
`output/staged_reconstruction/monst3r_geometry/adjacent_pair_confidence.png`.
It confirms early/middle confidence support is sparse; its bright late regions
do not constitute a depth-accuracy result. Hashes are in the receipts.

**No PIDG depth targets or training updates were produced.** No gate was
relaxed, no later frames selected to stand in for the whole sequence, and no
completeness result is claimed. This rejects the registered adjacent-pair pilot,
not the whole paper. A different pair graph or known-camera constrained global
alignment would require a separate protocol and actual qualification before
training. I9's original sources, controls and failed admission are unchanged.

Reproduction:

1. `python -m scripts.prepare_monst3r_assets --download-source --download-weights`
   fetches pinned, hash-checked official assets into dedicated project folders.
2. `python -m scripts.prepare_monst3r_pair_vendor` creates the documented import-only adapter.
3. `python -m scripts.qualify_monst3r_pairs prepare`, then `infer`, then `qualify`.
4. `python -m scripts.audit_monst3r_confidence` reports confidence failure and renders evidence.

Commands preserve existing receipts; use a new explicitly versioned pilot to
repeat a completed experiment. The first network attempt failed reading the
Windows certificate store; the successful retry used certifi with TLS
verification retained. The first loader attempt exposed tied-weight metadata;
no inference happened until strict loading succeeded. Qualification also had
an insufficient-support abort and a NumPy-boolean JSON-reporting error; original
logs are preserved, with successful final receipts distinct from those logs.
