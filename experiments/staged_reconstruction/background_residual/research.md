# Frozen-background residual neutralization: a controlled continuation

Primary source: DeGauss, ICCV2025,
<https://arxiv.org/html/2503.13176v1> and
<https://openaccess.thecvf.com/content/ICCV2025/html/Wang_DeGauss_Dynamic-Static_Decomposition_with_Gaussian_Splatting_for_Distractor-free_3D_Reconstruction_ICCV_2025_paper.html>.
The paper uses separate foreground/background representations, composition
masks and background brightness control. The relevant idea is coordinating
which branch explains appearance residuals. We do not reproduce its learned
masks, brightness field, density/reset schedule or full system here.

Our prior diagnostic found that static-region losses can send substantial
gradients to dynamic canonical positions even though S1 parameters are frozen.
Current combined-render attenuation at three probes is small, while dynamic-only
images remain broken. This supports isolating optimization coupling rather than
assuming static occlusion is the main problem. There is no causal percentage.

## Intervention

At a training camera, render frozen S1 alone as S. Let G be actual RGB and M
the original conditional blue visible-static observation rule. Candidate uses

    G_neutral = (1-M) G + M stop_gradient(S).

Only the full-image appearance target and the existing extra static-region term
use this reference. Foreground L1 uses actual G; union coverage and original
physics are unchanged. Static masks are computed from G, never from a changed
target. The same native renderer still jointly depth-sorts both populations.
An overlay that changes a visible blue pixel relative to S remains penalized;
we do not relabel an occluded/unknown3D region as empty. Actual known empty
background and observed yellow foreground retain their original supervision.
SSIM neighborhoods couple pixels, so only the direct foreground L1 gradient,
not the entire foreground gradient including SSIM, is claimed unchanged.

This is a controlled image-space reference experiment, not a claim that S1
geometry or color is GT. It may preserve static inaccuracies and suppress real
dynamic lighting effects on static objects. True train/val image evaluation and
the original static preservation requirement remain necessary and unchanged.

## Matched protocol

Two new arms start from the exact user-selected S3-U final6k, including Adam,
camera stack, Python/NumPy/CPU/CUDA RNG and physical neighbor caches. Both
continue2000 updates at native1600, with original physics at absolute6000+k.
No old initialization is regenerated, no material/instance count assumed,
no existing trainer/source/protocol is edited. This isolates continuation-time
static RGB residuals; it cannot rule out initialization or earlier-training
damage. Disabled wrapper calls the original loss verbatim.

Before formal training: synthetic reference/gradient tests, native flags-off
render/loss/gradient parity, finite original-physics smoke,100-step smokes in
both arms, independent candidate restarts at40 and80 and full-state audit.
Then evaluate validation30 and the original fixed6 cameras x9 times. A passing
screen still requires6x200 actual image inspection. No test RGB is used for
iterative selection. Original completeness, alpha and PSNR criteria persist;
metrics cannot certify intact geometry.
