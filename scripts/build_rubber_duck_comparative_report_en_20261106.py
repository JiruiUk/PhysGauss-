"""English edition of the audited report; experiment assets are read-only."""
from pathlib import Path
import json
import re
from xml.sax.saxutils import escape
import numpy as np
import build_rubber_duck_comparative_report_20261106 as r
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

ROOT = r.ROOT
STEM = 'RubberDuck_Comparative_Experiment_Report_EN_20261106'
r.PDF = ROOT / 'output/pdf' / (STEM + '.pdf')
r.TMP = ROOT / 'tmp/pdfs/rubber_duck_comparison_en_20261106'
r.FONT = Path('C:/Windows/Fonts/arial.ttf')
r.BOLD = Path('C:/Windows/Fonts/arialbd.ttf')
CW = r.CW
add, h, page, table, note, img = r.add, r.h, r.page, r.table, r.note, r.img


def charts(ev, controller):
    plt = r.plt
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    names = ['L4 Temporal memory', 'L5 Surface support', 'L6 Revocable springs', 'L8-R Relative order']
    control = np.array([31.636682, 31.223379, 31.938599, ev['validation']['means']['rgb_control']['PSNR']])
    candidate = np.array([31.319321, 31.859706, 31.589028, ev['validation']['means']['ordinal_depth']['PSNR']])
    yy = np.arange(4)
    ax.hlines(yy, control, candidate, color='#CAD5D9', lw=4)
    ax.scatter(control, yy, s=55, c=r.TEAL, label='Matched control in each trial', zorder=3)
    ax.scatter(candidate, yy, s=55, c=r.AMBER, label='Candidate in each trial', zorder=3)
    ax.axvline(31.994673, color=r.NAVY, ls='--', lw=1, label='J6 parent')
    ax.set(yticks=yy, yticklabels=names, xlabel='Mean PSNR over all 30 validation images (dB)')
    ax.set_ylim(3.55, -.5); ax.grid(axis='x', alpha=.15)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=3, frameon=False, fontsize=7.5)
    for i, (a, b) in enumerate(zip(control, candidate)):
        ax.text(min(a, b)-.025, i+.22, f'{b-a:+.3f} dB', fontsize=8, color=r.MUTED)
    comparison = r.savefig('matched_PSNR_comparison_en', fig)
    fig, ax = plt.subplots(figsize=(8.5, 2.7)); x = np.arange(3); bw = .23
    for i, (name, vals, color) in enumerate([
        ('L7-P (different original split)', [7, 7, 10], r.MUTED),
        ('Expanded anchors, per frame', [12, 9, 5], r.TEAL),
        ('Expanded anchors, joint', [17, 19, 3], r.AMBER)]):
        bars = ax.bar(x+(i-1)*bw, vals, bw, label=name, color=color)
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set(xticks=x, xticklabels=['Early 0-39', 'Middle 40-79', 'Late 80-119'],
           ylabel='Qualified views / 40 per period'); ax.set_ylim(0, 24)
    ax.legend(ncol=3, loc='upper center', fontsize=7.1, frameon=False)
    ax.grid(axis='y', alpha=.12)
    calibration = r.savefig('calibration_coverage_en', fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.8), sharey=True)
    for ax, (arm, label) in zip(axes, [('rgb_control', 'RGB control'), ('ordinal_depth', 'Relative-order candidate')]):
        rows = [dict(completed_updates=0, active=19, suspended=11, revoked=0)] + controller['arms'][arm]['refreshes']
        xx = [q['completed_updates'] for q in rows]; bottom = np.zeros(len(rows))
        for key, label2, color in [('active', 'Active', r.TEAL), ('suspended', 'Suspended', '#E5BA69'), ('revoked', 'Revoked', '#AAB6BC')]:
            vals = [q[key] for q in rows]
            ax.bar(xx, vals, 70, bottom=bottom, label=label2, color=color); bottom += vals
        ax.set(title=label, xlabel='Completed training updates', xticks=xx, ylim=(0, 33))
        ax.tick_params(axis='x', labelsize=8); ax.grid(axis='y', alpha=.1)
    axes[0].set_ylabel('Eligibility of the fixed 30 pairs')
    axes[1].legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.25), fontsize=8, frameon=False)
    return comparison, calibration, r.savefig('revocable_controller_en', fig)


def visuals():
    base = 'output/staged_reconstruction/RubberDuck_relative_order/v1'
    current = {}
    for cid in [86, 188]:
        cells = []
        for t in [2, 100, 200]:
            for model, label in [('J6_parent', 'J6'), ('rgb_control', 'RGB control'), ('ordinal_depth', 'Relative order')]:
                cells.append((f't{t:03d} / {label}', r.crop(f'{base}/c{cid:03d}_t{t:03d}_{model}_jointRGB_native.png', cid)))
        current[cid] = r.panel(f'fixed_{cid}_three_times_en', cells, 3, (560, 370))
    cells = []
    for cid, t in [(188, 100), (86, 200)]:
        for kind, label in [('dynamicRGB', 'Dynamic RGB'), ('dynamicAlpha', 'Dynamic alpha'), ('dynamicD-alpha', 'Dynamic D/alpha')]:
            cells.append((f'c{cid} t{t} / {label}', r.crop(f'{base}/c{cid:03d}_t{t:03d}_ordinal_depth_{kind}_native.png', cid)))
    decomposition = r.panel('dynamic_support_evidence_en', cells, 3, (560, 390))
    cells = []
    methods = [('RubberDuck_relative_order', 'J6_parent', 'J6'),
               ('RubberDuck_temporal_memory', 'temporal_memory', 'L4 Memory'),
               ('RubberDuck_surface_support', 'surface_support', 'L5 Structure'),
               ('RubberDuck_revocable_springs', 'revocable_spring', 'L6 Springs'),
               ('RubberDuck_relative_order', 'ordinal_depth', 'L8-R Order')]
    for kind, label in [('jointRGB', 'RGB'), ('dynamicAlpha', 'Dynamic alpha')]:
        for folder, model, name in methods:
            cells.append((f'{name} / {label}', r.crop(f'output/staged_reconstruction/{folder}/v1/c188_t100_{model}_{kind}_native.png', 188)))
    historical = r.panel('historical_same_view_en', cells, 5, (420, 340))
    sheet = r.source_image(f'{base}/observed_train_index119.png')
    assert sheet.size == (800, 1946)
    cells = []
    for i, label in enumerate(['Observed training RGB', 'J6', 'RGB control', 'Relative order']):
        y = 62+i*478
        cells.append((label, sheet.crop((0, y, 800, y+450)).crop((120, 35, 670, 420))))
    observed = r.panel('observed_trajectory_119_en', cells, 4, (420, 300))
    return current, decomposition, historical, observed


def footer(canvas, doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#D7E1E5')); canvas.setLineWidth(.5)
    canvas.line(42, 32, r.W-42, 32)
    canvas.setFont('ZH', 7.3); canvas.setFillColor(colors.HexColor(r.MUTED))
    canvas.drawString(42, 20, 'PhysGauss / Rubber Duck experiment comparison | 2026-11-06 | English edition')
    canvas.drawRightString(r.W-42, 20, f'{doc.page:02d}')
    if doc.page > 1:
        canvas.setFont('ZH', 7)
        canvas.drawRightString(r.W-42, r.H-23, 'Observations, inference and geometric ground truth are reported separately')
    canvas.restoreState()


def build():
    r.register(); r.styles()
    for name, style in r.STYLES.items():
        style.wordWrap = None
        style.allowWidows = 0; style.allowOrphans = 0
    for name, size, leading in [('body', 9.4, 14.1), ('small', 8.3, 11.6), ('cap', 7.9, 10.7),
                               ('h1', 20, 25), ('h2', 11.4, 16), ('table', 8.35, 11.5),
                               ('tablehead', 8.35, 11.5), ('formula', 9.1, 15), ('kicker', 8.2, 11.8)]:
        r.STYLES[name].fontSize = size; r.STYLES[name].leading = leading
    b = 'experiments/staged_reconstruction/'
    assessments = {name: r.source_text(b+folder+'/v1/assessment.md') for name, folder in [
        ('J6', 'RubberDuck_I10_joint'), ('L4', 'RubberDuck_temporal_memory'), ('L5', 'RubberDuck_surface_support'),
        ('L6', 'RubberDuck_revocable_springs'), ('L7-P', 'RubberDuck_pseudo_depth'), ('L7-J', 'RubberDuck_joint_scale'),
        ('L8', 'RubberDuck_ordinal_depth'), ('L8-R', 'RubberDuck_relative_order')]}
    base = b+'RubberDuck_relative_order/'
    protocol = r.read(base+'protocol.json'); ev = r.read(base+'v1/evaluation.json')
    ctrl = r.read(base+'v1/controller_audit.json'); pairs = r.read(base+'v1/pairs_audit.json')
    final = r.read(base+'v1/final_review.json'); audit = r.read(base+'v1/independent_audit.json')
    validation = r.read(base+'v1/final_validation.json'); r.read(base+'v1/videos.json')
    native = r.read(base+'v1/native_check.json'); recovery = r.read(base+'v1/recovery.json')
    r.read(base+'v1/training_receipt.json')
    train = r.read('data/derived/kinematic_resplit/training/transforms_train.json')
    val = r.read('data/derived/kinematic_resplit/transforms_val.json')
    assert len(train['frames']) == 120 and len(val['frames']) == 30
    assert len({q['time'] for q in train['frames']}) == 120
    assert len({str(q['transform_matrix']) for q in train['frames']}) == 120
    means = ev['validation']['means']; usage = ctrl['arms']['ordinal_depth']
    assert usage['positive_auxiliary_steps'] == 20 and usage['used_pair_observations'] == 74
    assert all(q['passed'] for q in [pairs, ctrl, audit, validation, native, recovery])
    for path in ['variants/staged_reconstruction/i10_relative_order/core.py',
                 'variants/staged_reconstruction/i10_ordinal_depth/core.py',
                 'variants/staged_reconstruction/i10_relative_order/runtime.py', base+'refresh_repair.md']:
        r.source_text(path)
    for name, value in [('L4', '31.319321'), ('L5', '31.859706'), ('L6', '31.589028')]:
        assert value in assessments[name], (name, value)
    comparison, calibration, controller = charts(ev, ctrl)
    fixed, decomposition, historical, observed = visuals()

    page(1, 'Rubber Duck<br/>Reconstruction Experiments', 'TECHNICAL COMPARISON / TRAINING, RELATIONS AND COMPLETENESS')
    add('From temporal memory, local structure and springs to pseudo-depth calibration and relative-order training', 'h2')
    add('Report date: 2026-11-06 | Main scope: J6 and L4-L8-R | English edition 1.0', 'small')
    note('Current conclusion', 'The relative-order candidate and its RGB control each completed 600 updates. The number of order-consistent relations did not increase, and fragmentation and trailing artifacts did not consistently improve in fixed views. Retain J6; all new weights and negative results are preserved.')
    table(['Model / trial', 'All-30 validation PSNR', 'Measured outcome'], [
        ['J6 parent', f"{means['J6_parent']['PSNR']:.3f} dB", 'Retained working model; off-trajectory defects remain'],
        ['RGB continuation, 600 steps', f"{means['rgb_control']['PSNR']:.3f} dB", 'Matched control'],
        ['Relative order, 600 steps', f"{means['ordinal_depth']['PSNR']:.3f} dB", '-0.101 dB versus control; visual gates failed']], [155, 114, CW-269])
    img(fixed[188], 'Cover example: camera 188, three times and three models, using identical crops of native renders. See page 15.', maxh=195)
    add('Evidence scope: creating this report did not add training or alter frozen results. Relative order and D/alpha are not independent geometric ground truth. The parent model\'s historical color-conditioned lineage remains disclosed. [S1-S8]', 'small')

    page(2, 'What this report establishes', 'FINDINGS AND READING GUIDE')
    table(['Question', 'Answer supported by the evidence'], [
        ['Did tracking / temporal constraints do nothing?', 'No. L4 reduced reprojection error for bound memories, but covered only 22 rows and mostly early observations. This did not establish whole-object completeness.'],
        ['Did connectivity, thickness or springs help?', 'Local proxies responded: L5 reduced connection gaps and L6 reduced mean strain. Global RGB, area / coverage tails and off-trajectory shape did not improve together.'],
        ['Can pseudo-depth directly supervise absolute depth?', 'L7-P and L7-J failed the original scale / dynamic-coverage gates; neither launched Gaussian training. Limited calibration cannot be extrapolated across the image.'],
        ['Is learning front-to-back order sufficient?', 'L8-R did train, but fit agreement stayed 9/30 and held agreement stayed 19/32. Whole-duck / environment 3D relations are still not established.'],
        ['Should training duration or weights simply increase?', 'The evidence does not support that next step. Broader temporal coverage and reliable relation-to-surface-node correspondence should be checked before preregistering one fixed candidate.']], [145, CW-145])
    h('Suggested reading order')
    add('Pages 3-4 define the data and comparisons. Pages 5-9 cover temporal memory, structure, springs and scale audits. Pages 10-13 explain relative-order training and numerical results. Pages 14-17 show native-image comparisons; pages 18-20 discuss interpretation, engineering checks and sources.')
    note('Three evidence types must remain distinct', '1. Training-trajectory RGB tests image fit. 2. Data inference and parent-model proxies test consistency of assumptions. 3. Independent geometric GT tests actual 3D correctness. These Rubber Duck experiments do not provide the third type for fixed-time, multiview dynamic geometry.')
    add('"Not promoted" means an experiment completed without replacing the retained working model. It does not mean the model was never updated or its new weights were not saved. [S1-S8]', 'small')

    page(3, 'Data, cameras and evidence', 'SHARED EXPERIMENT CONDITIONS')
    table(['Item', 'Verified meaning'], [
        ['Target dataset', 'The PIDG Rubber Duck benchmark. No new physical-camera capture was performed for this work.'],
        ['Training observations', '120 RGB images, 120 distinct exact times and 120 camera transforms. Simultaneous multiview observations: 0.'],
        ['Ordinary RGB training', 'Native 1600 x 900; 0.8 L1 + 0.2 DSSIM. Original RGBA is composited on black; alpha is not introduced as a new label.'],
        ['Validation', 'All 30 validation images at native 1600 resolution. This is separate from the auxiliary loss\'s held set.'],
        ['Fixed views', 'Camera IDs 86 / 188; time numbers 2 / 100 / 200, with normalized t = (number - 1) / 199. IDs are not training-list indices.'],
        ['Unified dynamic pool', '15507 dynamic Gaussian rows and a static library of 4126 points. These are model capacity, not a duck count.'],
        ['Known and unknown', 'Insufficient visibility, binding or boundary evidence remains unknown. Darkness or residuals do not justify labeling background / dynamic or deleting points.']], [128, CW-128])
    h('Seeing every surface over time does not automatically determine a full 3D model')
    add('Camera changes and object motion are coupled across time. Fusion requires a persistent correspondence between the surface seen now and the surface seen earlier, followed by correct motion compensation. Similar 2D appearance alone does not provide simultaneous multiview depth intersection or reliable front / back surface pairing.')
    add('J6 displacement and D/alpha can support hypotheses, but cannot independently prove those same hypotheses. Lower pixel error can coexist with flattened sheets, strands and dark yet opaque support.')
    add('Training metadata and the validation list were checked read-only during report generation. Test data were not used to select training points. [S1, S8, S15]', 'small')

    page(4, 'Lineage and fair comparisons', 'MODEL AND EXPERIMENT FAMILY')
    note('What is J6?', 'J6 combines I10 dynamic state with the user-accepted J4 static library, then compares frozen static parameters against trainable static opacity / SH for 2000 steps per arm. Joint J6 reached 31.994673 dB validation PSNR, +0.593549 dB over its matched control. The RGB gain was retained, while cleanup / completeness gates still failed. The combination start was 29.068226 dB and the frozen-static control reached 31.401124 dB; joint SSIM was 0.980425.')
    table(['Experiment', 'Steps / arm', 'Candidate adds', 'Control'], [
        ['L4 Memory', '2000', 'Measured cross-window UV memory and two-frame projected displacement', 'RGB'],
        ['L5 Support', '2000', 'Connectivity, local support thickness / area and observed-footprint contribution', 'RGB + same L4 mechanism'],
        ['L6 Springs', '2000', 'Local length strain with shared fixed k = 1', 'RGB + same L4 mechanism'],
        ['L7-P / L7-J', '0', 'Pseudo-depth qualification and joint scale audits', 'Read-only inference; no training arms'],
        ['L8 Intervals', '0', 'Calibrated absolute-depth interval separation', 'No fit pairs; no training'],
        ['L8-R Order', '600', 'Temporally stable order from raw relative inverse depth', 'RGB']], [88, 54, 220, CW-362])
    add('L4 / L5 / L6 / L8-R each start from the same complete J6, including two Adam states, RNG and remaining camera stack. Budgets and control losses differ; L5 and L6 controls also follow different CUDA numerical trajectories. Compare each matched increment, rather than ranking all candidates as equal-budget repeated seeds.')
    add('These are separate branches from J6, not sequential L4-to-L5-to-L6-to-L8-R continuation. The historical 40-step CUDA trajectory failure remains recorded. Single matched trials have no repeated-seed confidence intervals. [S1-S8]', 'small')

    page(5, 'L4: what temporal memory constrained', 'MEASURED CROSS-FRAME SURFACE MEMORY')
    h('Method: real observation memory, not synthesized unseen views')
    add('730 bidirectional RGB tracklets formed 641 memory units; 89 fit overlap links produced 75 long memories. Requirements included mutual-nearest matching, overlap RMS at most 0.75 px, maximum at most 1.5 px and ambiguity margin at least 0.25 px. Odd held frames only checked selected links. Ultimately 22 passed canonical binding and native own-row visibility, spanning 16-20 frames.')
    add('The candidate retrieves up to 4 real fit keyframes. Current projection, keyframe projection and two-frame projected displacement of shared canonical nodes receive gradients, with weight 0.01. Known per-frame cameras prevent treating raw image displacement as world motion. Unobserved frames and occlusion gaps receive no invented targets.')
    table(['Evidence', 'Measurement', 'Interpretation'], [
        ['Actual coverage', '22 rows, 372 observations; time thirds 356 / 16 / 0', 'Mostly early local memory, not whole-duck surface area'],
        ['Additional times', '18 memories add 76 measured observations outside the root window', 'More information, without proof of newly complete surfaces'],
        ['Held reprojection', 'J6 0.860543; control 0.895909; candidate 0.561905 px', 'Bound local correspondence improves; held still receives RGB training'],
        ['Actual loss usage', '216 / 2000 steps; 4734 keyframe-pair applications', 'Not every step or every surface receives supervision'],
        ['All-30 val PSNR', 'Control 31.636682; candidate 31.319321 dB', '-0.317361 versus control; -0.675352 versus J6; both gates fail']], [106, 202, CW-308])
    note('Established and unestablished', 'Gradients and improved reprojection confirm an effect on bound local tracks. They do not establish whole-object connectivity, thickness or off-trajectory shape. Fixed views 86 / 188 still show fragments and high-alpha trails.')
    add('Qualification refreshes every 250 steps: unknown suspends, 2 contradictions revoke, 2 passes restore. No new Transformer / RNN, point growth or unseen-surface triangulation was introduced. [S2]', 'small')

    page(6, 'L5: connectivity and surface support', 'LOCAL STRUCTURE CONSTRAINTS')
    add('Both 2000-step arms start from J6 and share L4 temporal memory. The candidate adds four averaged structural terms with total weight 0.01 and at most 3 qualified support groups per step. This is neither object segmentation nor true body-thickness supervision.')
    table(['Component', 'Implementation', 'Limitation'], [
        ['Connection / collapse', '2-sigma ellipsoid gap along the center line; half-parent-length lower bound', 'A parent reference can preserve old incorrect support'],
        ['Support thickness', 'Smallest principal-axis width of covariance plus center distribution, held within 0.8-1.25 times parent', 'Not physical front-to-back thickness or wall thickness'],
        ['Tangent area / scale', 'Product of two largest axes in the same range; maximum point scale at most 1.25 times', 'No guarantee of closure or missing-surface generation'],
        ['Observed footprint', 'Native depth-sorted group contribution at least 0.8 times parent', 'Own visible-support proxy, not independent GT']], [98, 215, CW-313])
    table(['Fixed 360 patch-frame metric', 'Temporal control', 'L5 candidate'], [
        ['Mean connection penalty', '0.044638', '0.018335 (-58.92%)'],
        ['Thickness outlier patch-frames', '2', '3'], ['Area outlier patch-frames', '19', '13'],
        ['Contribution below 0.8 times parent', '96', '67'],
        ['All-30 val PSNR', '31.223379', '31.859706 (+0.636327 dB)']], [214, 139, CW-353])
    add('Initial support: 60 patches, 275 distinct dynamic rows and 337 within-patch edge applications; start thirds 59 / 1 / 0. The 1.77% row coverage is not surface area. Structure was used on 166 steps, with 1174 patch and 5527 edge applications. Middle / late times and unrepresented surfaces remain poorly covered.')
    add('The candidate is -0.134967 dB versus J6 and passes both RGB non-degradation gates, but fixed-view fragmentation and dark high-alpha tails do not consistently disappear. A smaller local gap does not prove better complete geometry. Parent contribution exists for 359 / 360 patch-frames; the remaining unknown is not counted as zero error. [S3]', 'small')

    page(7, 'L6: benefits and costs of springs', 'SHARED FIXED-STIFFNESS GEOMETRIC REGULARIZATION')
    add('The trial again starts from J6 with a matched temporal control; it does not stack L5 thickness / area / footprint losses. Each reliable edge uses the median parent length over 6 fit frames as a fixed reference, shared k = 1 and total weight 0.01. No mass, contact, external force or material identification is added.')
    add('strain = (l(t) - l_ref) / l_ref<br/>E = sqrt(1 + strain<super>2</super>) - 1', 'formula')
    table(['Proxy over the fixed initial denominator', 'Control', 'Spring candidate'], [
        ['Mean absolute strain', '0.108432', '0.092034 (-15.12%)'],
        ['Mean spring energy', '0.011789', '0.008677 (-26.40%)'],
        ['Most negative edge strain', '-0.411848', '-0.695519'],
        ['Area outlier patch-frames', '9', '16'], ['Contribution below 0.8 times parent', '67', '76'],
        ['Validation PSNR', '31.938599', '31.589028 (-0.349571 dB)']], [214, 139, CW-353])
    h('Why lower mean strain did not yield a complete shape')
    add('Average length regularization mainly reduces local stretch. It creates no surfaces and does not ensure that area or compression tails improve. Connections still cover only 60 patches / 275 rows, with start thirds 59 / 1 / 0. The parent median is not a physical rest length and may suppress genuine flexible deformation.')
    add('Springs were used on 178 / 2000 steps, with 1439 patch and 6608 edge applications. PSNR drops 0.405645 dB versus J6; both 0.3 dB control / parent gates fail. Strands, fragments and dark support persist, so the candidate was not promoted.')
    note('What this says about spring-mass constraints', 'The experiment tests a local spring geometry regularizer, not full mass-spring dynamics. Its result does not support increasing stiffness or adding mass / unknown forces to compensate for correspondence and coverage gaps. [S4]')

    page(8, 'L7-P: pseudo-depth qualification', 'SCALE, MOTION COMPENSATION AND BOUNDARIES / NO GAUSSIAN UPDATES')
    add('The fixed official Depth Anything V2 Small processes all 120 training RGB images at observation resolution 400 x 225, with 518 preprocessing and FP32 evaluation. It is neither changed nor fine-tuned. Finite predictions on black background are not automatically surfaces.')
    h('Scale and motion checks')
    add('Triangulated anchors from stationary-compatible RGB tracks fit a positive-affine inverse-depth model per frame. Spatial fit / check separation and two-fit-subset stability must pass; only interpolation within the fit convex hull is allowed. Parent J6 rendered depth is not scale-fitting GT.')
    add('z(u) = 1 / (a r(u) + b), a &gt; 0<br/>X_hat(t) = T_camera(t)<super>-1</super> [z K<super>-1</super> p] - displacement_J6(t)', 'formula')
    add('Camera backprojection first removes the change of camera coordinates; J6 time-varying displacement is then subtracted to check a track\'s world consistency. This is parent-conditioned compensation, not proof of binding or true motion. Even frames fit, odd frames check; each side requires at least 4 valid observations.')
    table(['Qualification measure', 'Observed', 'Original gate'], [
        ['Calibrated views', '24 / 120', 'At least 30'], ['Early / middle / late', '7 / 7 / 10', 'At least 8 per third'],
        ['Continuous-motion checks passed', '0 tracks', 'At least 30; at least 5 per third'],
        ['Distinct dynamic rows', '0', 'At least 25'],
        ['Native-qualified track observations', '45 / 1464 locally usable', 'At least 4 fit and 4 check per track']], [187, 165, CW-352])
    add('There are 122 native-qualified tracks x 12 observations = 1464. At most 7 / 12 depths are valid for any track, so no track reaches the numerical motion-compensation test. This does not show tracking or compensation to be ineffective. Of 53196 boundary observations, 42276 remain unknown. The limited known subset does not show higher boundary failure and cannot establish occlusion as the sole cause. [S5]', 'small')

    page(9, 'L7-J: joint scale and more anchors', 'DATA-INFERENCE AUDIT / NO GAUSSIAN UPDATES')
    add('1981 new full-image texture seeds yield 963 RGB-qualified tracks and 265 stationary-compatible tracks; adding 362 old tracks gives 627. Cross-window duplicates form 507 groups, split as whole groups into fit / check and two fit subsets. Per-frame 4 px deduplication leaves 5351 anchor observations. Static points are not selected by blue color.')
    add('120 normalized affine inverse-depth models are fitted jointly with fixed Huber error, a 0.05 second-time-difference prior, positive-slope bounds and tiny ridge. Smoothness is a prior, not evidence of correct scale; held data can only check, not adjust coefficients.')
    img(calibration, 'Figure 1. Each third contains 40 frames. Joint and per-frame fits with the same expanded anchors are comparable. Original L7-P uses a different split and is not a strict single-factor control. [S5-S6]', maxh=190)
    table(['Metric', 'Same anchors, per frame', 'Joint across time'], [
        ['Qualified views / 120', '26', '39'], ['Locally usable dynamic observations / 1464', '16', '75'],
        ['Tracks reaching the motion test / 122', '0', '2'], ['Motion checks passed', '0', '1 (5 fit depth proposals)']], [231, 126, CW-357])
    add('Only track 052_020, dynamic row 6886, passes: 11 local observations and held compensation residual median 1.842%, P90 3.623%. These are not independent GT. Only 3 / 40 late views qualify, and dynamic track / row gates fail; the 5 depth proposals are not trained. Potential fit-convex-hull image area rises 6.846% to 12.884%, not accurate surface coverage. [S6]', 'small')

    page(10, 'L8 and L8-R test different hypotheses', 'INTERVAL FAILURE PRESERVED / SEPARATE SCALE-FREE CANDIDATE')
    table(['Design', 'L8 Interval order', 'L8-R Raw temporal order'], [
        ['Order source', 'Separation of full / split calibrated absolute-depth intervals', 'Small raw relative inverse-depth patch intervals'],
        ['Metric scale', 'Requires stable calibrated intervals', 'No absolute-scale fitting'],
        ['Dynamic tracks', 'L7-J tracks passing metric checks', 'Original 122 T4 native-qualified tracks'],
        ['Cross-frame checks', 'Interval reliability and spatial support', 'At least 4 even same-sign and 4 odd checks'],
        ['Training / diagnostic pairs', 'Fit 0 / 94; held 1 / 43', 'Fit 30; held 32; all hypotheses retained'],
        ['Actual training', '0 steps', '600 steps per arm']], [104, 203, CW-307])
    h('Why L8 did not train')
    add('Among 94 fit proposals, 77 have overlapping absolute intervals and 17 have weak raw order, leaving 0 training pairs. The single held pair remains diagnostic and is not moved into training. This outcome retains the original gates; permission to try training does not manufacture valid targets.')
    h('How L8-R separately tests relative front-to-back relations')
    add('A scale-free single candidate was separately preregistered. It skips metric intervals but requires raw order to persist through the observation window, remain locally smooth, and agree across fit / check times. It neither rewrites the failed L8 result nor relaxes old global geometry admission gates. The authorization is for a local exploratory trial, not automatic inference of whole-duck relations.')
    note('Identity and order remain separate', 'Endpoints are a measured UV bound to a dynamic row and a stationary-compatible RGB-anchor UV. They carry no duck ID or environment semantics. A maximum of 4 pairs per image is not an object count. [S7-S8]')

    page(11, 'L8-R: targets and the loss', 'CORE MECHANISM OF THE COMPLETED TRAINING TRIAL')
    add('Let S be image q90 - q10 of raw relative inverse depth. A strong sign requires both 5 x 5 endpoint patches to have range at most 0.1 S, endpoint distance at least 8 px, and interval separation at least 0.1 S. Larger inverse depth means nearer. Missing, boundary or contradictory evidence remains unknown.')
    add('Within 12 frames of a dynamic-track / anchor-group hypothesis, at least 4 even fit frames must support one sign s, and at least 4 odd check frames must confirm it. Check can reject but cannot choose a replacement sign. Whole anchor groups are separated; stable track / group ordering caps each image at 4 pairs, with fixed confidence 0.5.')
    table(['Full construction denominator', 'Fit library', 'Diagnostic held library'], [
        ['Group hypotheses', '4540', '2953'], ['Temporal support passed', '41', '27'],
        ['Frame proposals before cap', '193', '129'], ['Final fixed relation pairs', '30', '32'],
        ['Dynamic rows', '6498, 15419', '6498, 15419'],
        ['Frame indices', '8 even frames in 100-114', '8 odd frames in 101-115']], [205, 153, CW-358])
    add('At native 400 joint-render resolution, bilinearly sample D and alpha, then set z = D / alpha. Require positive finite depth, alpha &gt; 0.5 and current dynamic-row reprojection within 4 px. The signed log-depth gap is:')
    add('g = s [log(z_anchor) - log(z_dynamic)]<br/>e = max(0, log(1.02) - g)<br/>L_order = (0.001 / N) sum [0.5 * valid * (sqrt(1 + (e / 0.05)<super>2</super>) - 1)]', 'formula')
    add('N is the original fixed per-frame proposal denominator, not the shrinking valid subset. Target sign, confidence and qualification are detached. Gradients pass only through current native D/alpha to dynamic canonical / grid / deform parameters. s = +1 means the dynamic endpoint should be nearer. No absolute depth or large depth gap is specified. [S8-S9]', 'small')

    page(12, 'Revocable eligibility, sparse usage', 'ACTUAL TRAINING AND QUALIFICATION HISTORY')
    img(controller, 'Figure 2. The fixed 30 pairs refresh every 100 steps. Initially 19 active / 11 suspended; after the candidate\'s step-500 refresh: 7 active / 2 suspended / 21 revoked. Final eligibility carries over from that refresh; there is no extra step-600 check. [S10]', maxh=207)
    table(['Check / training item', 'Fixed rule or measured result'], [
        ['Own contribution and RGB', 'Native single full-scene depth sorting; own-row patch contribution share at least 0.01. At dynamic UV, observed / rendered patch NCC at least 0.6 and both standard deviations at least 2 on a 255 scale.'],
        ['Revoke / restore', 'Unknown immediately suspends; 2 contradictions revoke; 2 passes restore. Current projection, alpha and positive depth are also checked every step.'],
        ['Training budget', '600 native-1600 RGB updates per arm. Only the candidate adds a 400-resolution order render. Fixed Adam epsilon 1e-15.'],
        ['Actual order usage', '20 / 600 nonzero auxiliary steps; 74 valid pair-observation applications at indices 102 / 104 / 106 / 112.'],
        ['Frozen state', 'Static model / Adam, physics / Adam / graph caches and topology. No L4 / L5 / L6 losses are stacked.']], [124, CW-124])
    add('A 1% own-contribution threshold is only a visibility gate, not evidence that the row dominates joint depth. Other Gaussians along the ray can substantially affect D/alpha. This limits the link between image order and a specific surface-node relation.')
    add('20 / 600 is about 3.33%. The 600 RGB updates must not be described as 600 depth-constraint updates. Losses and gates stayed fixed; weights and duration were not automatically increased after results. [S8-S10]', 'small')

    page(13, 'Results: order counts did not improve', 'MATCHED NUMERICAL RESULTS AND CROSS-TRIAL CONTEXT')
    table(['Model', 'PSNR dB', 'SSIM', 'L1'], [[name, f"{means[key]['PSNR']:.6f}", f"{means[key]['SSIM']:.6f}", f"{means[key]['L1']:.8f}"]
        for key, name in [('J6_parent', 'J6 parent'), ('rgb_control', 'Current RGB control'), ('ordinal_depth', 'Current relative order')]], [158, 112, 112, CW-382])
    table(['Order diagnostic (not GT)', 'J6', 'RGB control', 'Candidate'], [
        ['Fit agreement / fixed 30', '9/30', '9/30', '9/30'], ['Fit valid / fixed 30', '28/30', '28/30', '28/30'],
        ['Held agreement / fixed 32', '19/32', '19/32', '19/32'], ['Held valid / fixed 32', '32/32', '32/32', '32/32']], [220, 97, 97, CW-414])
    add('Candidate PSNR is -0.100921 dB versus control and -0.164675 dB versus J6. Both maximum-drop 0.3 dB gates pass, which establishes only limited non-degradation. Mean fit signed log-gap changes -0.026816 to -0.024436; held changes 0.022797 to 0.024123. No pair changes its prior-order side or validity.')
    img(comparison, 'Figure 3. Each candidate is compared with its own matched control. L4 / L5 / L6 use 2000 steps; L8-R uses 600. These are neither repeated seeds nor a ranking against one shared control. [S2-S4, S8]', maxh=195)
    add('Across all 122 tracks / 1464 observations, held reprojection is 0.911032 px for the candidate, 0.909937 for control and 0.888095 for J6. Binding does not improve globally. Held excludes auxiliary targets but still receives RGB training; it is not independent geometric generalization. [S8]', 'small')

    page(14, 'Camera 86: early, middle and late', 'IDENTICAL CROPS OF NATIVE 1600-RESOLUTION RGB')
    img(fixed[86], 'Figure 4. Columns: J6 / current RGB control / relative-order candidate. Rows: time numbers 2 / 100 / 200. Identical native-RGB crops, without repainting or retouching. [S11]', maxh=445)
    h('Observations')
    add('Early surfaces remain loose with visible gaps. A recognizable middle-time outline does not remove posterior particle chains. All three late-time models retain stretched particle sheets, fragmented regions and dark support. The candidate does not consistently join fragments into complete objects.')
    add('The static layer is fully frozen, so the blue support structure is unchanged. Different dynamic occlusion is not a static reconstruction improvement. There is no independent RGB / geometry GT for these fixed-camera cross-time combinations; the images support defect comparison, not calibrated geometric error. [S11]', 'small')

    page(15, 'Camera 188: strands and gaps persist', 'IDENTICAL CROPS OF NATIVE 1600-RESOLUTION RGB')
    img(fixed[188], 'Figure 5. Columns: J6 / current RGB control / relative-order candidate. Rows: time numbers 2 / 100 / 200. Sampling matches Figure 4; each camera uses its own fixed crop. [S11]', maxh=445)
    h('Observations')
    add('Early duck bodies remain fragmented. Middle-time images show laterally extended particle bands and strands. Late RGB looks more recognizably duck-like, but that does not establish removal of dark high-alpha structures. Differences are mainly local details, without consistent completion.')
    add('Unchanged local order and broadly similar renders agree in this trial. Neither observation disproves all relative priors or RGB solvability. Current supervision touches only 2 dynamic Gaussian rows at late times. [S8, S11]', 'small')

    page(16, 'Darker RGB is not removed geometry', 'JOINT REVIEW OF DYNAMIC RGB, ALPHA AND D/ALPHA')
    img(decomposition, 'Figure 6. Relative-order candidate: c188 / t100 above, c86 / t200 below. Columns: dynamic RGB / dynamic alpha / dynamic D/alpha. Depth colors reuse the same parent-static range within each camera. Black low-alpha regions are missing, not zero error. [S11]', maxh=310)
    add('Dark lateral bands in RGB still show clear alpha and D/alpha support. A darker color does not establish deletion of incorrect geometry. Conversely, high alpha plus dark color does not justify classifying background or deleting points.')
    table(['Dark RGB and alpha &gt; 0.5, six fixed images', 'Pixel count', 'Comparison'], [
        ['J6', '107498', 'Parent reference'], ['Current RGB control', '121627', 'Above J6'],
        ['Relative-order candidate', '113170', '-6.95% versus control; +5.28% versus J6']], [CW-223, 99, 124])
    add('Defined only for evaluation: every dynamic RGB channel &lt; 0.05 and alpha &gt; 0.5. Denominator: 1440000 pixels per image, 8640000 pixel applications total. This is not independent-GT area or color-based training classification. Both visual cleanup and completeness gates fail. [S8, S11]', 'small')

    page(17, 'Earlier branches in the same view', 'DIFFERENT MECHANISMS, PERSISTENT DEFECTS')
    img(historical, 'Figure 7. Same world region at camera 188 / t100. Columns: J6, L4, L5, L6, L8-R. RGB above, dynamic alpha below. Different budgets and losses prevent a strict equal-budget ranking. [S2-S4, S8, S11]', maxh=245)
    h('Shared observations across trials')
    add('Temporal memory, local structure, springs and relative depth change local details, yet fragmentation and lateral high-alpha support remain. Their proxies differ: pixel reprojection, parent support geometry, parent median edge length and monocular-prior order. Self-consistency of any one proxy does not validate an entire duck surface.')
    img(observed, 'Figure 8. Observed training image at list index 119 and the three original-trajectory renders, rearranged from existing images. Appearance is cleaner here. This is a training camera / time observation, not independent GT for fixed cameras 86 / 188 across time. [S11]', maxh=135)
    add('Figure 7 shows candidates, not their matched controls; matched numerical increments are on page 13. Figures 7 and 8 have different time / camera conditions and illustrate evaluation scope. Subtracting them would not measure geometric error.', 'small')

    page(18, 'What is missing, and what to test next', 'TECHNICAL LIMITATIONS AND ACTIONABLE CHECKS')
    table(['Supported limitation', 'Do not overinterpret', 'Priority check'], [
        ['Sparse temporal support: L4 / L5 / L6 are early-heavy; L8-R has only 2 late rows', 'This does not show that no other usable surfaces exist in the RGB sequence', 'Use fixed denominators by time, view and visible surface; expand reliable binding before training'],
        ['Uncertain 2D-track to canonical-node / patch correspondence', 'Row counts and graph links are not object identities', 'Keep revocable links; independently check correspondence across occlusion and surface membership'],
        ['Joint D/alpha mixes Gaussian contributions; own-share floor is only 1%', 'Order gradients need not modify the intended surface node', 'Report contribution dominance and surface association before designing a relation loss'],
        ['Parent reference length / thickness / depth may preserve wrong support', 'Better self-proxies do not prove real geometry', 'Add mutually exclusive observation checks; protect only evidenced surface relations'],
        ['No surface growth or reliable front / back pairing', 'More regularization cannot create missing surfaces automatically', 'Require testable evidence for new surfaces and thickness before imposing either']], [174, 155, CW-329])
    h('Recommended sequence, not an executed result')
    add('First audit surface nodes and visible contributions over broader times / views, retaining unknowns and original candidates. If evidence is sufficient, fix one node-level relation or local surface-fusion mechanism; declare the matched budget, thresholds, native-gradient and recovery checks, then run one trial. Stacking several losses at once would obscure attribution.')
    note('Current decision', 'Retain J6. Preserve the complete L8-R weights without promotion. Do not automatically extend training, raise weights, sweep parameters or alter gates. This negative result does not invalidate temporal information, all pseudo-depth, or RGB solvability. Physical mass, constitutive laws and forces were not added.')
    add('These limits follow from fixed denominators, native images and implementation behavior. Validation / test data did not select new training points. [S1-S11]', 'small')

    page(19, 'Engineering checks and retained failures', 'CORRECT EXECUTION AND GEOMETRIC SUCCESS ARE DISTINCT')
    table(['Verification', 'Completed in this trial'], [
        ['Tests / default off', '20 relevant tests. Native RGB / alpha / D exact equality and original gradient tolerances pass.'],
        ['Recovery contract', 'User-accepted exact restoration of full model, two Adam states and bindings, RNG, camera stack and physics graph. Adam updates are exact with the same native gradients.'],
        ['Gradients', 'Actual order loss is computable. Qualification-only forced violations verify finite nonzero canonical / grid / deform gradients; they are not training targets.'],
        ['Full replay', '1200 camera samples; exact native replay of 10 saved refresh models from steps 100-500; independent eligibility-transition checks.'],
        ['Artifacts', '180 image SHA / readability checks; all fields of 6 full-Gaussian PLYs plus 9 world-XYZ diagnostics; 2 H264 videos fully decoded, 200 frames each.'],
        ['Freeze / closure', '608 frozen-file SHA checks and 17 task-frozen Git raw-byte checks. pending:false; task lock released.']], [124, CW-124])
    h('First 100-step failure and independent repair')
    add('The first formal RGB control exited after 100 completed updates because stale measurement metadata collided with a new refresh field. No complete step-100 checkpoint existed, so partial state was not resumed. Failure stdout, initial state and progress are retained. Independent repair_run strips only old measurement metadata, then calls the frozen original transition logic.')
    add('After more than 600 supplemental state calls and revoke / restore checks, both arms again passed short training and independent recovery. Each then restarted from the original complete J6 for 600 steps. The final comparison contains 1200 updates, separate from the failed control\'s 100 and two qualification rounds. Original epsilon 1e-15, losses, gates and historical 40-step CUDA failure remain unchanged.')
    add('Commits: 5e1a0800 preregistration; aa62478d first qualification; 6b2d6f3c repair record; 4a8173d8 requalification; eb70f4e4 training; 53b7fbfc results; 9874a28d final validation. Browser playback was not tested. Manual review used original images and decoded contact sheets, not manual frame-by-frame playback. [S8-S14]', 'small')

    page(20, 'Sources, weights and review entry points', 'REPOSITORY EVIDENCE INDEX')
    add('Paths below are relative to F:/A_Edinburgh/PhysGauss. The companion manifest records SHA hashes for report sources and every source image. This English edition preserves the Chinese original and all historical experiment files.', 'small')
    refs = [
        ('S1', 'J6 method, lineage and results', 'RubberDuck_I10_joint/v1/assessment.md'),
        ('S2', 'L4 temporal method and results', 'RubberDuck_temporal_memory/v1/assessment.md'),
        ('S3', 'L5 structure method and results', 'RubberDuck_surface_support/v1/assessment.md'),
        ('S4', 'L6 spring method and results', 'RubberDuck_revocable_springs/v1/assessment.md'),
        ('S5', 'L7-P pseudo-depth qualification', 'RubberDuck_pseudo_depth/v1/assessment.md'),
        ('S6', 'L7-J joint scale', 'RubberDuck_joint_scale/v1/assessment.md'),
        ('S7', 'L8 zero-fit interval failure', 'RubberDuck_ordinal_depth/v1/assessment.md'),
        ('S8', 'L8-R report / protocol / evaluation', 'RubberDuck_relative_order/{protocol.json,v1/assessment.md,v1/evaluation.json}'),
        ('S9', 'Order implementation', 'variants/staged_reconstruction/{i10_relative_order,i10_ordinal_depth}/core.py'),
        ('S10', 'Usage and refresh audit', 'RubberDuck_relative_order/v1/{controller_audit.json,pairs_audit.json}'),
        ('S11', 'Native images, PLYs and videos', 'output/staged_reconstruction/RubberDuck_{relative_order,temporal_memory,surface_support,revocable_springs}/v1/'),
        ('S12', 'Independent artifact audit', 'RubberDuck_relative_order/v1/independent_audit.json'),
        ('S13', 'Recovery and native gradients', 'RubberDuck_relative_order/v1/{recovery.json,native_check.json}'),
        ('S14', 'Final state / validation / failure', 'RubberDuck_relative_order/{refresh_repair.md,v1/final_review.json,v1/final_validation.json}'),
        ('S15', 'Training / validation cameras', 'data/derived/kinematic_resplit/{training/transforms_train.json,transforms_val.json}')]
    for sid, label, path in refs:
        prefix = '' if path.startswith(('variants/', 'output/', 'data/')) else 'experiments/staged_reconstruction/'
        add(f'<b>[{sid}] {label}</b> '+escape(prefix+path), 'cap')
    h('Current checkpoint SHA-256')
    for label, value in [('J6', protocol['parent_sha256']), ('Current RGB control', ev['checkpoint_audits']['rgb_control']['checkpoint_sha256']),
                         ('Current order candidate', ev['checkpoint_audits']['ordinal_depth']['checkpoint_sha256'])]:
        add(f'{label}: {value[:32]}<br/>{value[32:]}', 'small')
    add('Both trial checkpoints are in runs/pidg_staged_reconstruction/RubberDuck_relative_order/v1/formal/{rgb_control,ordinal_depth}/checkpoint.pt. Full state is preserved; no automatic resumption or rerun. The J6 parent is runs/pidg_staged_reconstruction/RubberDuck_I10_joint/v1/formal/joint_depth_trust/checkpoint.pt.', 'cap')

    doc = r.ReportDoc(str(r.PDF), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=44,
                      title='Rubber Duck Reconstruction: Comparative Experiment Report',
                      author='PhysGauss experiment records',
                      subject='Temporal memory, local structure, springs, pseudo-depth and relative-order evidence', pageCompression=1)
    doc.build(r.story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(str(r.PDF)); texts = [q.extract_text() for q in reader.pages]
    assert len(reader.pages) == 20, (len(reader.pages), r.page_specs)
    assert len(reader.outline) == 20
    for i, (_, title) in enumerate(r.page_specs):
        assert title.replace('<br/>', '')[:12] in texts[i].replace('\n', ''), (i, title, texts[i][:150])
    assert all(len(t) > 80 for t in texts)
    fulltext = '\n'.join(texts)
    assert not re.search(r'[\u3400-\u9fff]', fulltext), 'Untranslated Chinese in PDF text'
    for token in ['31.829998', '31.930919', '31.994673', '9/30', '19/32', '608', '9874a28d']:
        assert token in fulltext, token
    original_pdf = ROOT / 'output/pdf/RubberDuck_实验比较技术报告_20261106.pdf'
    original_manifest = json.loads(original_pdf.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    assert r.digest(original_pdf) == original_manifest['sha256']
    assert r.sources == original_manifest['experiment_files_and_images'], 'Source set must be unchanged'
    for path, sha in r.sources.items():
        assert r.digest(ROOT/path) == sha
    manifest = dict(pdf=r.PDF.relative_to(ROOT).as_posix(), sha256=r.digest(r.PDF), pages=len(reader.pages),
                    authoring_source=Path(__file__).relative_to(ROOT).as_posix(), authoring_source_sha256=r.digest(__file__),
                    layout_helper='scripts/build_rubber_duck_comparative_report_20261106.py', layout_helper_sha256=r.digest(r.__file__),
                    translated_from=original_pdf.relative_to(ROOT).as_posix(), original_pdf_sha256=r.digest(original_pdf),
                    experiment_files_and_images=r.sources,
                    figures={name: dict(embedded_in_pdf=True, sha256=r.digest(path)) for name, path in r.figures.items()},
                    expected_titles=r.page_specs, no_training_or_frozen_result_mutation=True, language='en',
                    font_embedded='Arial TrueType regular/bold',
                    evidence_scope='J6, L4, L5, L6, L7-P, L7-J, L8 and L8-R; English translation; no new experiments')
    r.PDF.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('PDF_CREATED', r.PDF, 'PAGES', len(reader.pages), 'SOURCE_FILES', len(r.sources), flush=True)


if __name__ == '__main__':
    build()
