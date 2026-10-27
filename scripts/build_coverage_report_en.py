"""English companion to the verified 15-page Chinese coverage ablation report."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import pidg_ablation_report_layout as L

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
PDF = OUT / 'PIDG_Coverage_Ablation_Report_EN.pdf'
CN = OUT / 'PIDG_Coverage_Ablation_Report_CN.pdf'
ASSESS = ROOT / 'experiments/coverage_isolated/assessment.json'
ASSETS = OUT / 'coverage_assets'
PAGE_COUNT = 15
RECORD = ['# PIDG Coverage and Neighborhood Ablation Report', '']
METRICS = ['PSNR', 'SSIM', 'LPIPS', 'MS-SSIM']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Page(L.Page):
    def __init__(self, number, title, subtitle=''):
        RECORD.extend(['', '## ' + title, ''])
        super().__init__(number, 'C0-C5 Coverage Ablation', title, subtitle)

    def text(self, text, style='body', gap=10):
        RECORD.append(text + '\n')
        super().text(escape(text).replace('\n', '<br/>'), style, gap)

    def heading(self, text):
        RECORD.extend(['', '### ' + text, ''])
        super().heading(text)

    def callout(self, text):
        RECORD.append(text + '\n')
        super().callout(escape(text).replace('\n', '<br/>'))

    def table(self, rows, widths, **kwargs):
        RECORD.append('| ' + ' | '.join(map(str, rows[0])) + ' |')
        RECORD.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
        RECORD.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows[1:])
        RECORD.append('')
        super().table(rows, widths, **kwargs)

    def image(self, filename, height, caption):
        RECORD.extend(['', f'![{caption}](coverage_assets/{filename})', ''])
        super().image(ASSETS / filename, height, caption)

    def path(self, path):
        RECORD.append('`' + str(path) + '`\n')
        L.Page.text(self, L.path_markup(str(path)), 'path', gap=7)


def setup():
    # Font aliases are local to this process; shared layout/source files stay intact.
    for name, filename in [('CN', 'arial.ttf'), ('CN-Bold', 'arialbd.ttf'),
                           ('Arial', 'arial.ttf'), ('PathFont', 'consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + filename))
    pdfmetrics.registerFontFamily('CN', normal='CN', bold='CN-Bold', italic='CN', boldItalic='CN-Bold')
    for key, style in L.styles.items():
        style.wordWrap = None
        if key != 'path':
            style.fontName = 'CN'
        if key == 'body':
            style.fontSize, style.leading = 10, 14.5
        elif key in ('small', 'caption'):
            style.fontSize, style.leading = 8.4, 11.8
        elif key in ('table', 'thead'):
            style.fontSize, style.leading = 8.2, 11.6
    L.TOTAL_PAGES = PAGE_COUNT
    L.page_records.clear()
    L.c = canvas.Canvas(str(PDF), pagesize=(L.W, L.H), pageCompression=1)
    L.c.setTitle('PIDG Coverage and Neighborhood Ablation Report: C0-C5')
    L.c.setAuthor('PhysGauss / PIDG Extension Project')
    L.c.setSubject('English companion: methods, results, mask color failure, and 2D alpha metrics')


def build(a):
    models = a['models']
    cw = L.CW
    p = Page(1, 'Better coverage, worse RGB reconstruction?',
             'Completed C0-C5 experiments and implementation analysis | 27 October 2026')
    p.callout('No added method improves all metrics over V3 in these runs. Mask supervision improves silhouette IoU but reduces RGB quality. Local affine regularization partly recovers the mask-related loss. Density-only stays closest to the reference, with small metric trade-offs and attribution confounds.')
    p.heading('What changes across the six models?')
    p.table([['Model', 'Mask L1 weight', 'Local affine weight', 'Coverage additions'],
             ['C0 / V3', '0', '0', 'Off'], ['C1', '0.1', '0', 'Off'],
             ['C2', '0.1', '0.0001', 'Off'], ['C3', '0.1', '0.0001', 'On'],
             ['C4', '0', '0.0001', 'Off'], ['C5', '0', '0', 'On']], [72,120,140,cw-332])
    p.heading('Matched comparison protocol')
    p.text('Scene: Rubber Duck. All arms use W256_D0, the V3 RGB objective and a shared frame-index time axis. Each runs for 50k steps from the same complete random step-0 state; C0 reuses the completed control. Isotropic and velocity-trajectory consistency weights are zero. The 120 train / 30 validation / 50 test images have disjoint timestamps. This is interpolation at interleaved unseen times, not future forecasting.')
    p.text('The main tables use final iteration 50000 for every arm. Four additional best snapshots are selected using validation metrics only. Outputs are 1600 x 900 PNGs. These additions change losses or Gaussian allocation, not MLP width; Gaussian count remains an optimization outcome.')
    p.text('Important scope correction: the scene contains several ducks and obstacles. C2/C3/C4 use unsegmented spatial neighborhoods without real object IDs. Cross-object connectivity has not been ruled out.', 'small')
    p.finish()

    p = Page(2, 'Final RGB metrics: C0 has the highest PSNR',
             'Fixed 50-frame test set; higher PSNR/SSIM/MS-SSIM and lower LPIPS are better')
    p.table([['Model', *METRICS], *[[m['id'], f"{m['scores']['PSNR']:.4f}",
                *[f"{m['scores'][k]:.6f}" for k in METRICS[1:]]] for m in models]], [48,98,120,120,cw-386])
    p.heading('Separate isolated and conditional effects')
    rows = [['Contrast', 'Interpretation', 'PSNR delta / dB', 'Better frames / 50']]
    for key, label in [('C1-C0','Mask only'), ('C4-C0','Affine only'), ('C5-C0','Density only'),
                       ('C2-C1','Affine after mask'), ('C3-C2','Add density to both')]:
        c = a['comparisons'][key]
        rows.append([key,label,f"{c['metric_delta']['PSNR']:+.4f}",c['per_frame']['PSNR']['improved_frames']])
    p.table(rows, [75,150,135,cw-360])
    p.text('C5 slightly improves SSIM and LPIPS over C0, while PSNR and MS-SSIM are slightly worse. This is a trade-off, not a uniform gain. C2 improves over C1 by about 1.41 dB but remains about 2.57 dB below C0. Recovering part of a degradation does not establish superiority to the reference.')
    p.text('C3 beats C2 on 25/50 frames, yet its mean PSNR is 1.29 dB lower. The final eight test timestamps, original IDs 169-197, lose 5.57 dB on average and contribute about 69% of the mean degradation. Failure is concentrated in particular time intervals.', 'small')
    p.text('These are observed single-run differences. Trajectories already differ before the new terms activate, so the final deltas are not isolated causal effect estimates.', 'small')
    p.finish()

    p = Page(3, 'Alpha metrics reveal a different trade-off',
             'Silhouette accuracy, interior coverage and RGB reconstruction are distinct objectives')
    rows = [['Model','Foreground PSNR','Mask IoU','Interior low coverage %','Background false coverage %']]
    for m in models:
        q, d = m['coverage']['mean'], m['alpha_diagnostics']['mean']
        rows.append([m['id'],f"{q['foreground_psnr']:.3f}",f"{q['mask_iou']:.4f}",
                     f"{100*q['interior_hole_rate']:.5f}",f"{100*d['background_false_positive_rate']:.2f}"])
    p.table(rows,[40,105,85,133,cw-363])
    p.image('frame153_alpha.png',220,'Full-frame alpha at original frame 153. White indicates high opacity; all panels share the same 0-1 display range.')
    p.text('C0 combines a very low interior hole rate with poor silhouette IoU. It retains substantial high alpha in GT background regions: black RGB is not equivalent to transparency. Mask supervision removes much of this false coverage, but does not establish correct interior color, position or occlusion order.')
    p.text('An opaque black pixel may also look like a hole. Color or occlusion errors of this kind are not counted as low-alpha holes, so foreground RGB must be examined separately.', 'small')
    p.text('IoU thresholds both alphas at >0.5. Interior low coverage counts predicted alpha<0.5 inside GT alpha>0.95 eroded by 3 pixels. Background false coverage counts predicted alpha>0.5 where GT alpha<=0.5. Foreground PSNR uses GT alpha>0.5. Results are frame-wise means; exact formulas and denominators are on page 15.', 'small')
    p.finish()

    p = Page(4, 'Audit: differences precede density additions',
             'A shared random initialization does not imply bitwise identical GPU training')
    p.image('validation_curves.png',178,'Validation every 1k steps. Improvement slows at later steps, with runs settling at different reconstruction levels. Validation curves alone do not establish overfitting.')
    p.image('prefix_differences.png',164,'Shading marks the period before the first additions. Coverage sampling starts at 16k; the first clones are added at 18k, after that step\'s validation.')
    prefix = a['pre_activation_validation_psnr_differences']
    p.table([['Validation PSNR difference','15k: before additions','50k: final'],
             *[[key,f"{prefix[key]['15000']:+.4f}",f"{prefix[key]['50000']:+.4f}"] for key in ['C5-C0','C3-C2']]], [175,175,cw-350])
    p.text('C3 trails C2 by about 0.80 dB before additions. C5 instead leads C0 by about 0.54 dB at 15k. Final differences therefore cannot all be attributed to cloning. The exact nondeterminism source is unresolved; custom CUDA accumulation and amplification by densification thresholds are candidates.', 'small')
    p.text('The affine comparisons have a similar limitation: at 3k, before first activation, C4-C0 and C2-C1 already differ by about -0.08 and +0.07 dB. Shared-state branching and repeated seeds are needed for stronger attribution.', 'small')
    p.finish()

    p = Page(5, 'Matched RGB views and per-frame performance',
             'Original frame 153 / export 00038; the previously discussed difficult frame, with one GT-defined crop')
    p.image('frame153_rgb.png',288,'GT and C0-C5 share identical crop coordinates. The final panel shows C1 mean absolute RGB error with a fixed 0-0.25 range. This frame is diagnostic, not a representative sample estimate.')
    p.image('test_per_frame.png',166,'PSNR for all 50 test timestamps, plotted against original frame ID. Connecting segments do not imply test metrics were computed for intervening train/validation frames.')
    p.text('C1 has a clear color error in this frame. C2/C3 retain local shape or occlusion defects. Alpha agreement alone does not ensure correct color. The visible multi-object structure also shows why a distance gate cannot establish object membership.', 'small')
    p.text('Dark gaps on the blue obstacles can arise from transparency, wrong color or occlusion. Compare RGB with the alpha panels rather than interpreting every dark patch as missing Gaussian coverage.', 'small')
    p.finish()

    p = Page(6, 'Method 1: foreground coverage supervision',
             'Used by C1/C2/C3; soft silhouette supervision, distinct from random frame removal')
    p.heading('Targets and loss integration')
    p.text('Read the original A channel from the 120 training RGBA images, resize with bicubic interpolation to camera resolution, and divide by 255. The original RGB loader and black-background compositing remain unchanged. This is neither a SAM mask nor a flow-validity mask. Training-membership checks prevent validation/test masks from entering the loss; held-out alpha is used only for evaluation.')
    p.callout('L_mask = (1 / HW) sum_p |A(p,t) - M(p,t)|\nL_total = L_V3 + 0.1 L_mask + scheduled L_affine\nA: rendered accumulated alpha. M: detached GT alpha.\nDensity additions are a discrete allocation rule, not another differentiable loss.')
    p.text('L_V3 retains pixel L1, SSIM and existing physics/grid regularizers, plus LPIPS with weight 0.02. LPIPS runs every four steps with interval compensation by four. Mask loss runs every step from step 1, without foreground/background balancing, edge weighting, delay or ramp. Isotropic and velocity-trajectory consistency weights are zero.', 'small')
    p.heading('Which variables receive gradients?')
    p.text('Visible contributors with active gradients propagate alpha errors into opacity, position, scale and rotation. After deformation warm-up, dynamic centers and covariance also connect to the deformation network. Mask loss has no color target: in the alpha compositing graph, its direct derivative with respect to SH color coefficients is zero. RGB losses still train SH.')
    p.text('Nevertheless, changing geometry, opacity and occlusion changes which SH colors contribute and how RGB gradients are distributed. The mask constrains a 2D silhouette and opacity, not the unique depth, object correspondence, ordering or color.')
    p.heading('Implementation')
    p.text('coverage_tuning/data.py: read_alpha; model.py: mask_loss; runtime.py: coverage_loss; code/train_pidg.py: merge into the total before loss.backward(). Setting mask_weight=0 disables this term.', 'small')
    p.finish()

    p = Page(7, 'Method 2: weak local affine consistency',
             'Used by C2/C3/C4; allows real stretching rather than fixing every neighbor distance')
    p.heading('Neighborhoods and relative coordinates')
    p.text('Select 128 deterministic anchors that cycle with iteration. Find up to eight nearest canonical neighbors per anchor and retain edges shorter than 0.1 world units. X denotes detached canonical coordinates and y(t) the current deformed center. Relative offsets cancel translation.')
    p.callout('D_i[j,:] = X_j - X_i; E_i[j,:] = y_j(t) - y_i(t)\nB_i = pinv(D_i) E_i; E_fit = D_i B_i\nq_i = mean_j ||D_i[j,:]||^2\nL_affine = mean_i [mean_j ||E_i[j,:] - E_fit[j,:]||^2 / max(q_i, 1e-8)]')
    p.text('Means use valid edges only. A neighborhood needs at least four edges and a smallest singular value exceeding 1e-4 times its largest. Pseudoinverse rtol is 1e-4. D, its pseudoinverse and the normalization are detached; E and the fitted B remain differentiable with respect to deformation. This regularizer updates the trajectory, not canonical coordinates directly.', 'small')
    p.heading('Schedule, permitted motion and limitations')
    p.text('Evaluate every 10 steps after 3k, ramping over 2k steps. The evaluated-step multiplier is 1e-4 x 10 x ramp, compensating for sparse evaluation. Rotation, uniform stretch and shear are allowed. There is no det(F)=1, volume-conservation or constitutive-stress target. B is a fitted local map, not a validated physical deformation gradient F.')
    p.text('Non-affine bending, contact boundaries and relative object motion can also create residuals. Real training supplies no object_ids, so the radius gate cannot guarantee a shared surface or object. The original single_object declaration was not scene-verified and was corrected in interpretation. C4 acknowledges this limitation but still lacks identity isolation.')
    p.text('Implementation: model.py neighborhoods/local_affine_loss and runtime.py trajectory/ramp. Tests cover affine invariance, non-affine gradients and degenerate neighborhoods; they do not prove that real-scene adjacency is correct.', 'small')
    p.finish()

    p = Page(8, 'Method 3: late coverage-guided densification',
             'Used by C3/C5; clone existing contributors supported by deficits at several training times')
    p.heading('Accumulate pixel deficits onto existing Gaussians')
    p.callout('d(p,t) = 1[p in eroded GT foreground] x 1[A(p,t) < 0.7]\n             x max(0, 0.95 - A(p,t))\nS_i = sum of d(p,t) over pixels/times listing i as a contributor\nV_i = number of distinct deficit-supporting training frames\nOnly parents with V_i >= 3 and S_i > 0 are eligible.')
    p.text('Erode GT alpha>0.95 by three pixels. Sample every 20 steps from 16k through 35,999 on a four-pixel grid, reading at most 20 actual footprint contributors per pixel. Each valid listed ID receives the same deficit; the current score is not normalized by RGB/alpha contribution weight. Repeated samples at one timestamp increase score but count as one time vote.', 'small')
    p.heading('Cloning, inheritance and budget')
    p.text('At 18k, 20k, ..., 36k, rank eligible parents by S_i and clone at most 512, with a total cap of 200,000 points. Reset scores and votes after each window. Add zero-mean canonical Gaussian jitter with per-axis standard deviation 0.5 x min(parent scale, 0.02). The value 0.02 is not a hard displacement-norm limit.')
    p.text('Children inherit SH, scale, rotation, particle/velocity index and dynamic flag. Clamp initial child opacity to [0.01,0.1]. Keep all parents, with no extra late pruning or opacity reset. Conventional gradient-based densification/pruning still stops at 15k. Additions run after the optimizer step to avoid discarding gradients during parameter replacement.')
    p.heading('What can this strategy recover?')
    p.text('It strengthens weak existing footprints using multi-time evidence. Pixels without any contributor have neither a parent nor reliable depth and are skipped. It cannot seed arbitrary empty geometry, and inherited wrong SH/positions/trajectories may propagate defects. C3/C5 add 879/717 points respectively, a conservative increment rather than global reconstruction.')
    p.text('Implementation: runtime.py observe_coverage/after_update and density.py CoverageAccumulator. Scores and per-point/time votes are stored in a checksum-bound checkpoint sidecar. Selection and cloning run under no_grad.', 'small')
    p.finish()

    build_remaining(a, models, cw)
    L.c.save()


def build_remaining(a, models, cw):
    """Pages 9-15 follow the same section order as the Chinese companion."""
    p = Page(9, 'Why can mask supervision change color?',
             'Mechanisms implied by this implementation, not a contributor-level causal diagnosis')
    p.heading('1. Color and alpha share visibility weights')
    p.callout('a_i(p) = opacity_i x projected_footprint_i(p)\nT_i(p) = product_{j<i} (1 - a_j(p)); w_i(p) = T_i(p) a_i(p)\nC(p) = sum_i w_i(p)c_i(view) + T_end(p)b\nA(p) = 1 - T_end(p)\nAway from sorting/threshold changes: dC/dc_i = w_i; dL_mask/dc_i = 0.')
    p.text('Color c_i comes from per-Gaussian spherical harmonics (SH), not an RGB output of the deformation MLP. The deformation branch predicts center transforms, rotation and scale, with no independent time-conditioned color residual. Although mask loss does not directly update SH, its updates to opacity, position and covariance alter weights w_i, blended colors, occlusion and subsequent SH gradients. Sorting changes are nonsmooth and are not fully described by these derivatives.')
    p.heading('2. A correct silhouette can have wrong contributors')
    p.text('Many depth, color and Gaussian configurations can yield A near one. A wrongly colored foreground contributor can hide correct points while maintaining high alpha; the mask cannot distinguish this. Occluded points have smaller w_i and weaker RGB correction signals. On black backgrounds, near-zero RGB can result from transparency or opaque, nearly black contributors. Alpha supervision removes some RGB-equivalent solutions and redistributes geometry and color.')
    p.heading('3. Early supervision and discrete allocation may amplify errors')
    p.text('Mask loss starts at step 1, while d_xyz, d_rotation and d_scaling remain zero for iterations below 3000. Different motion timestamps are sampled against a temporarily static cloud, so silhouette targets may conflict. Mask gradients also enter screen-space positions used by conventional densification before 15k. Consequently, mask supervision can alter point allocation as well as opacity. Both pathways exist in the code; neither has been established as the main cause of this color failure.')
    p.finish()

    p = Page(10, 'Color failure: evidence and open explanations',
             'Left duck at original frame 153; identical GT-defined region and pixel selection for every model')
    p.image('frame153_color_alpha_zoom.png',230,'Top: RGB. Bottom: alpha. Shared crop x=620:735, y=350:465. C1 remains wrongly colored at high opacity, so increasing alpha alone cannot correct this defect.')
    case = a['mask_color_case']
    rows = [['Image','Mean R / G / B','Mean alpha','RGB MAE']]
    for label, stats in case['models'].items():
        rows.append([label,' / '.join(f'{v:.3f}' for v in stats['mean_rgb_at_gt_selected_pixels']),
                     f"{stats['mean_alpha_at_gt_selected_pixels']:.4f}",f"{stats['rgb_mae_at_gt_selected_pixels']:.4f}"])
    p.table(rows,[55,230,110,cw-395])
    p.text(f"Statistics use {case['selected_gt_pixels']:,} GT pixels satisfying R>100, G>90 and B<130 inside the fixed crop, in 8-bit RGB. This is post hoc localization, not training supervision or a guarantee of shared material-point correspondence. Across all 50 frames, C1 improves silhouette IoU but loses about 4.23 dB of foreground PSNR.", 'small')
    p.heading('What does this frame establish?')
    p.text('C1 has zero red at all 3,850 selected pixels, while mean G/B remain close to GT. This makes channel clamping a more specific lead than generic color mixing. However, an 8-bit PNG cannot establish a negative pre-clamp SH value. The mask has not been shown to directly turn yellow into green, and weight 0.1 has not been proven excessive. Other candidates include visibility mismatch, conflicting static warm-up targets and competing gradients.', 'small')
    p.text('Proposed checks: mask on/off from one full state; freeze geometry or SH separately; log RGB/mask gradient norms and cosine similarity on shared parameters; compare delayed/ramped and weaker mask weights; inspect contributor color, depth and identity in this region. These checks have not been run.', 'small')
    p.finish()

    p = Page(11, 'A specific lead: clamping of SH color channels',
             'Zero red agrees with a zero-gradient source branch, but contributor-level evidence is still required')
    p.heading('Facts verified in the local implementation')
    p.text('These runs use convert_SHs_python=False. In the CUDA rasterizer, forward.cu computes SH color, adds 0.5, then applies max(result,0) channel-wise and records whether result<0. In backward.cu, dL_dRGB is multiplied by zero for clamped channels before calculating SH and viewing-direction gradients.')
    p.callout('q_i,k = SH_i,k(view) + 0.5; c_i,k = max(q_i,k, 0)\nIf q_i,k < 0, then dc_i,k / dq_i,k = 0.\nFor that Gaussian, view and channel, the color-evaluation gradient into SH is zero. Increasing RGB/perceptual loss weight alone cannot cross this clamped path.')
    p.heading('Why could this explain a yellow-to-green appearance?')
    p.text('In the fixed region, GT mean R is 0.733, whereas C1 has R=0 at all 3,850 selected pixels, nearly preserved G/B, and alpha about 0.964. If dominant contributors have negative red SH outputs before clamping, the result can stay opaque while losing red light and appearing green. A mask target contains no red-channel information and cannot detect that error from alpha alone.')
    p.text('A testable chain is: mask changes geometry, visibility and gradient allocation; RGB optimization or a changing view direction moves some channels into the clamped region; local zero gradients then impede recovery. We did not log the first negative transition, pre-clamp values, contributor weights or clamp flags here. The chain is therefore a hypothesis, not a confirmed explanation for this run or for every green artifact.')
    p.heading('How to confirm or reject the hypothesis')
    p.text('First perform a read-only diagnostic render of floating-point RGB and record visible contributors, raw SH values, clamp flags and weights. Measure the weighted share of negative-red contributors. Tiny positive values can also quantize to zero in 8-bit PNGs and must be ruled out. Then compare delayed mask supervision and a smooth positive color parameterization under matched initialization/state branching. These are future experiments; neither activation changes nor retraining were performed for this report.')
    p.text('Zero gradient is local to this sample/channel/path, not necessarily permanent: other views or times, geometry changes, contributor replacement and optimizer momentum can still change the outcome.', 'small')
    p.text('Source: third_party/PIDG/submodules/flow-based-diff-gaussian-rasterization/cuda_rasterizer/forward.cu, lines 65-70; backward.cu, lines 29-34 and 48-97. Source hashes are preserved in the Chinese report provenance referenced by this companion.', 'small')
    p.finish()

    p = Page(12, 'Interpretation: separate evidence from hypotheses',
             'Synthesis of the method details and color diagnostics; no single cause has been proven')
    p.heading('1. Mask supervision changes the admissible reconstruction')
    p.text('Evidence: C1 increases silhouette IoU from C0\'s 0.2064 to 0.9764, but lowers full-image PSNR by about 3.98 dB and foreground PSNR by about 4.23 dB. Degradation is not confined to the background.')
    p.text('Interpretation: black-background RGB can be matched using opaque, nearly black Gaussians. Alpha supervision suppresses that solution and jointly changes scale, opacity, position and deformation gradients. This improves the silhouette objective but may disrupt a favorable RGB allocation. Weight 0.1 applies globally from step 1, while roughly 90% of pixels are background. Gradient balance deserves testing, but no weight sweep establishes that 0.1 is too large.')
    p.heading('2. Neighborhood regularization can stabilize and restrict')
    p.text('Evidence: C2 improves RGB and interior coverage relative to C1, whereas C4 with affine-only regularization remains below C0. Local cooperation may stabilize incoherent motion under mask supervision, while restricting genuine non-affine detail. Canonical Euclidean neighbors also lack object identities and may couple contacting objects. Actual cross-object edge counts have not been established without labels.')
    p.heading('3. More points do not guarantee correct geometry')
    p.text('Evidence: C3 adds 879 points and C5 adds 717, ending at 70,258 and 70,848 points; C0 ends at 75,542. Conventional densification has already produced different populations, so an additions arm can still have fewer final points than the reference. The rule clones contributors near weakly covered footprints. It cannot infer depth in contributor-free gaps, and inherited color, scale or trajectories may perpetuate mistakes.')
    p.text('C3\'s decline relative to C2 cannot be fully attributed to additions because the gap predates them. C5 is close to or slightly above C0 only on selected metrics; this is not evidence of a comprehensive, repeatable reconstruction improvement.', 'small')
    p.finish()

    p = Page(13, 'Next steps: improve attribution before tuning',
             'Recommended experiments only; this report does not launch new training')
    p.table([['Priority','Actionable comparison','Question addressed'],
             ['1','Branch density on/off from one complete 15k state; retain a 50k total budget and repeat at least three seeds.','Remove pre-addition divergence and test whether a roughly 0.3 dB difference repeats.'],
             ['2','Compare weaker and delayed/ramped mask supervision from random initialization; log RGB/mask gradient norms on shared parameters.','Test strength and timing rather than attributing failure to capacity.'],
             ['3','Use known object IDs or reliable adjacency filters; propagate IDs through cloning and inspect contact/boundary regions.','Exclude cross-object constraints; color or proximity alone does not establish identity.'],
             ['4','Jointly inspect alpha false positives/negatives, foreground RGB and local errors; track new-point contribution and coverage changes.','Avoid apparently good hole rates from opaque background and verify useful additions.']], [45,268,cw-313])
    p.heading('What should be retained now?')
    p.text('Keep C0 as the main RGB reference. Retain C5 as a lightweight density candidate, not a default improvement. Mask supervision substantially improves silhouette accuracy and should remain tunable rather than being dismissed. Resolve adjacency reliability before deciding whether local affine regularization belongs in the final model.')
    p.heading('Limits of the conclusions')
    p.text('One scene and one seed provide no repeated-run confidence interval. Gaussian count is an outcome rather than fixed capacity. The split measures interleaved temporal interpolation, not long-gap reconstruction or forecasting. The test set has historical exposure and user-authorized early viewing in this study; it is not a completely untouched final test. No recovery claim is made for GT depth, velocity or force.', 'small')
    p.text('Training and validation use disjoint 120/30 images; the fixed 50-frame test set is used for evaluation. All four best snapshots are selected on validation. The main tables use final50k throughout to avoid retrospectively selecting snapshots using test results.', 'small')
    p.finish()

    p = Page(14, 'Absolute paths to renders and comparisons',
             'Each individual gallery contains 250 GT/render pairs: five snapshots x 50 test frames')
    for m in models:
        directory = Path(m['directory'])
        p.heading(m['id'] + ' / ' + m['name'])
        p.path(directory.parent.parent / 'render_comparisons' / m['name'] / 'index.html')
        p.path(directory / 'test/ours_50000/renders')
    p.text('GT and alpha are in sibling gt and alpha folders. The final50k pairs are in each gallery\'s pairs/50000 folder. Other tags: best, metric_ssim, metric_lpips and metric_ms_ssim.', 'small')
    p.heading('Combined GT + six-model gallery: 50 final50k images')
    p.path(ROOT / 'runs/pidg_coverage_comparison/final50000/index.html')
    p.text('Sources: results.json, per_view.json, coverage_metrics_test.json, metric_history.jsonl, metric_best.json, configs, initialization/export receipts, and runtime/model/density implementations. Full hashes and per-frame diagnostics are preserved in the provenance and experiments/coverage_isolated/assessment.json.', 'small')
    p.finish()

    p = Page(15, 'How are the two coverage percentages computed?',
             'Definitions for page 3; both measure 2D alpha coverage and lower is better')
    p.text('A_t(p) is rendered alpha; M_t(p) is GT alpha from the original RGBA frame. Both use the same 1600 x 900 grid and range 0-1. |S| denotes pixel count, not area in 3D.')
    p.heading('1. Interior low coverage: missing opacity inside the object')
    p.callout('I_t = {p: minimum GT alpha in the surrounding 7 x 7 window > 0.95}\nh_t = count(p in I_t with A_t(p) < 0.5) / max(|I_t|, 1)\nInterior low coverage % = 100 x (1/50) sum_t h_t')
    p.text('I_t is GT alpha>0.95 eroded with a square of radius three pixels, using zero outside the image. This excludes silhouette and semitransparent edges. The denominator is the eroded interior pixel count; the numerator counts alpha<0.5 there. It is not normalized by the full image or all foreground pixels.', 'small')
    p.heading('2. Background false coverage: opacity on true background')
    p.callout('B_t = {p: M_t(p) <= 0.5}\nf_t = count(p in B_t with A_t(p) > 0.5) / max(|B_t|, 1)\nBackground false coverage % = 100 x (1/50) sum_t f_t')
    p.text('The denominator is GT background count; the numerator is predicted-foreground count within it. This is FP/(FP+TN), not FP/(TP+FP), and it does not classify background by black RGB. M=0.5 belongs to background. Predicted A=0.5 is counted as neither low coverage nor false coverage.', 'small')
    p.heading('Worked pixel counts: original frame 153 / final50k')
    rows = [['Model','Interior: low/total','Frame %','Background: false/total','Frame %']]
    for m in models[:2]:
        row = m['alpha_diagnostics']['per_frame'][38]
        low, inside = row['interior_low_coverage_pixels'], row['interior_pixels']
        fp, bg = row['background_false_positive_pixels'], row['background_pixels']
        rows.append([m['id'],f'{low}/{inside}',f'{100*low/max(inside,1):.5f}',f'{fp}/{bg}',f'{100*fp/max(bg,1):.2f}'])
    p.table(rows,[40,130,70,177,cw-417])
    p.text('This example is one frame. Main-table values are the equal-weight mean of 50 per-frame fractions, multiplied by 100, not pooled pixel counts. For example, 0.00185% is a fraction of 0.0000185. Empty-region denominators are clamped to one and yield zero.', 'small')
    p.text('Evaluation reads 8-bit predicted alpha PNGs; GT alpha is bicubic-resized and scaled to 0-1. These thresholded diagnostics differ from training soft-alpha L1. Opaque color errors are not counted as interior holes. Broad opaque coverage can give low interior holes but high background false coverage. Neither metric establishes 3D watertightness.', 'small')
    p.text('Verification: all 300 predicted-alpha images reproduce the stored per-frame interior counts and fractions. Background counts are saved in assessment.json. Implementation: coverage_tuning/model.py interior/coverage_metrics and build_coverage_report.py alpha_diagnostics.', 'small')
    p.finish()


def main():
    a = json.loads(ASSESS.read_text(encoding='utf-8'))
    source = json.loads(CN.with_suffix('.sources.json').read_text(encoding='utf-8'))
    assert source['expected_pages'] == PAGE_COUNT and source['pdf_sha256'] == sha(CN)
    for relative, digest in source['sources'].items():
        assert sha(ROOT / relative) == digest, relative
    assert a['protocol']['snapshot'] == 'final50000' and len(a['models']) == 6
    setup()
    build(a)
    assert len(L.page_records) == PAGE_COUNT
    PDF.with_suffix('.md').write_text('\n'.join(RECORD), encoding='utf-8')
    used = [CN, CN.with_suffix('.sources.json'), ASSESS, Path(__file__),
            ROOT / 'scripts/pidg_ablation_report_layout.py', *sorted(ASSETS.glob('*.png'))]
    proof = {'created_utc': datetime.now(timezone.utc).isoformat(), 'pdf': str(PDF),
             'pdf_sha256': sha(PDF), 'language': 'en', 'expected_pages': PAGE_COUNT,
             'pages': L.page_records, 'sources': {str(p.relative_to(ROOT)): sha(p) for p in used},
             'chinese_companion_sha256': sha(CN), 'data_checks': a['protocol'],
             'coverage_metric_definitions': a['coverage_metric_definitions'],
             'translation_scope': 'All 15 sections; unchanged numerical metrics, formulas, figures and render paths'}
    PDF.with_suffix('.sources.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pdf': str(PDF), 'pages': L.page_records}, indent=2))


if __name__ == '__main__':
    main()
