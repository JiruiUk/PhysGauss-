"""Record the assistant's actual image inspection, not automatic visual scoring."""
from pathlib import Path
from variants.staged_reconstruction.prepare_data import ROOT,write,sha

B=Path(__file__).resolve().parent
O=ROOT/'output/staged_reconstruction/RubberDuck_J7_long_memory/v1'
paths=[]
for cid in [86,188]:
    for t in [2,100,200]:
        for kind in ['joint_comparison','decomposition','temporal_memory_jointRGB_native']:
            paths.append(O/f'c{cid:03d}_t{t:03d}_{kind}.png')
    paths.append(O/f'fixed_c{cid:03d}_parent_control_joint.decoded.png')
for index in [0,24,48,72,96,119]:paths.append(O/f'observed_train_index{index:03d}.png')
paths += [O/'memory_coverage.png',O/'c086_t200_RGB_alpha_D_focus.png',O/'c188_t100_RGB_alpha_D_focus.png']
assert len(paths)==29 and all(p.exists() for p in paths)
write(B/'v1/manual_review.json',dict(actually_viewed=True,reviewer='assistant inspected actual tool-returned images',
    images={p.relative_to(ROOT).as_posix():sha(p) for p in paths},image_count=len(paths),
    static_structure_preserved=True,dynamic_cleanup_and_completeness_improved=False,
    observations=[
        'Static RGB pillars/base bodies remain; broad static alpha and spikes persist.',
        'c86 early native RGB still has fragmented surface and isolated particles.',
        'c86 late dark dynamic high-alpha oval persists; memory support is visibly larger than matched control.',
        'c188 middle native RGB still fragments; alpha/D show long planar/triangular streaks.',
        'Observed trajectory often looks good; fixed camera across other times remains inconsistent.',
        'Decoded early/middle/late contacts agree with native comparisons; no browser playback was tested.',
        'Initial memory chart has no final-third observations; local track gain is not whole-surface reconstruction.'
    ],D_alpha_is_rendered_proxy=True,black_missing_is_not_zero_error=True,independent_geometry_GT=False,
    actual_human_user_approval_not_claimed=True,source_sha256=sha(Path(__file__))))
print('J8_ACTUAL29_IMAGE_REVIEW_RECORDED')
