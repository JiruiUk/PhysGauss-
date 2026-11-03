"""Independently check unchanged checkpoint/report bindings and label old images."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from variants.staged_reconstruction.prepare_data import ROOT, DATA, read, write, sha

BASE = ROOT/'experiments/staged_reconstruction/S1_joint_revocation_audit'
REPORT = BASE/'v1'
OUT = ROOT/'output/staged_reconstruction/S1_joint_revocation_audit/v1'


def main():
    assert not (REPORT/'independent_review.json').exists()
    protocol, result = read(BASE/'protocol.json'), read(REPORT/'formal.json')
    assert read(REPORT/'preflight.json')['passed']
    for receipt in ['source_freeze.json', 'repair_source_freeze.json']:
        for name, expected in read(REPORT/receipt).items():
            assert sha(ROOT/name) == expected, name
    for name, expected in result['imported_sources'].items():
        assert sha(ROOT/name) == expected, name
    assert sha(ROOT/protocol['checkpoint']) == protocol['checkpoint_sha256']
    state = torch.load(ROOT/protocol['checkpoint'], map_location='cpu', weights_only=False)
    digest = hashlib.sha256()
    for name, value in state['model'].items():
        array = value.detach().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    assert digest.hexdigest() == result['model_hash_before_and_after']
    assert not any('static' in group.get('name', '') for group in state['optimizer']['param_groups'])
    assert result['optimizer_updates'] == 0 and result['caller_RNG_unchanged_including_construction']
    meta = read(DATA/'transforms_train.json')
    assert [r['train_list_index'] for r in result['rows']] == protocol['fixed_train_list_indices']
    opacity = []
    for row in result['rows']:
        frame = meta['frames'][row['train_list_index']]
        assert row['frame'] == frame['file_path'] and row['time'] == frame['time']
        assert row['model_hash'] == digest.hexdigest() and row['caller_flags_RNG_and_leaf_grads_preserved']
        assert row['RGB_alpha_flags_on_off_exact']
        for group in row['gradient_groups'].values():
            assert group['finite'] and group['l2'] > 0
            assert group['positive']+group['negative']+group['zero'] == group['connected_values']
            assert group['connected_values'] <= group['parameters']
        grad = row['gradient_groups']['static_opacity']
        assert grad['parameters'] == len(state['model']['static._opacity'])
        opacity.append({'frame': row['frame'], **grad})
    first = read(REPORT/'preflight_0.json')['rows'][0]
    for group, values in first['gradient_groups'].items():
        actual = result['rows'][0]['gradient_groups'][group]
        for name, value in values.items():
            if name in ['l2', 'mean_abs']:
                assert np.isclose(value, actual[name], atol=protocol['independent_reload_gradient_atol'],
                    rtol=protocol['independent_reload_gradient_rtol'])
            else:
                assert value == actual[name]
    for name, expected in result['images'].items():
        assert sha(ROOT/name) == expected, name
        with Image.open(ROOT/name) as image:
            image.verify()
    assert len(result['images']) == 54
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(opacity)); bottom = np.zeros(len(opacity))
    for key, color, label in [('positive', '#d95f02', 'g>0: local descent reduces opacity logit'),
        ('negative', '#1b9e77', 'g<0: local descent increases opacity logit'), ('zero', '#777777', 'zero')]:
        values = np.asarray([r[key]/r['parameters']*100 for r in opacity])
        ax.bar(x, values, bottom=bottom, color=color, label=label); bottom += values
    ax.set_xticks(x, [Path(r['frame']).stem for r in opacity]); ax.set_xlabel('fixed observed training frame')
    ax.set_ylabel('fraction of all 4,126 static opacity parameters (%)'); ax.set_ylim(0, 103)
    ax.set_title('Legacy S3-U full-RGB gradients only: NO UPDATE, NO ownership/geometry GT')
    ax.legend(loc='lower right', fontsize=8)
    plot = OUT/'static_opacity_gradient_signs.png'; assert not plot.exists()
    fig.savefig(plot, dpi=150); plt.close(fig)
    fixed = Image.new('RGB', (1920, 615), '#171923')
    for col, cam in enumerate(protocol['fixed_native_cameras']):
        for row, frame in enumerate(protocol['fixed_native_frame_numbers']):
            with Image.open(OUT/f'fixed{cam}_t{frame:04d}.png') as im:
                fixed.paste(im.resize((960, 205), Image.Resampling.LANCZOS), (col*960, row*205))
    fixed_path = OUT/'fixed_old_model_review.png'; assert not fixed_path.exists(); fixed.save(fixed_path)
    observed = Image.new('RGB', (1280, 1230), '#171923')
    for row, index in enumerate(protocol['fixed_train_list_indices']):
        with Image.open(OUT/f'observed_index{index:03d}.png') as im:
            observed.paste(im.resize((1280, 205), Image.Resampling.LANCZOS), (0, row*205))
    observed_path = OUT/'observed_old_model_review.png'; assert not observed_path.exists(); observed.save(observed_path)
    review_images = {p.relative_to(ROOT).as_posix(): sha(p) for p in [plot, fixed_path, observed_path]}
    write(REPORT/'independent_review.json', {'passed': True, 'checkpoint_and_six_model_hashes_exact': True,
        'raw_RGBA_train_target_and_legacy_color_mask_lineage_disclosed': True,
        'six_fixed_train_bindings_and_gradient_sign_denominators_verified': True,
        'preflight_index0_aggregate_replay_with_original_tolerance': True,
        'all54_old_native_and_sheet_image_hashes_and_decode_verified': True,
        'optimizer_updates': 0, 'geometry_staticity_or_cleanup_proven': False,
        'static_opacity_gradients': opacity, 'review_images': review_images,
        'formal_sha256': sha(REPORT/'formal.json'), 'source_sha256': sha(Path(__file__))})
    print('J0_INDEPENDENT_REVIEW_PASSED', flush=True)
    for row in opacity:
        print(row['frame'], 'opacity+', row['positive'], 'opacity-', row['negative'], 'zero', row['zero'], flush=True)


if __name__ == '__main__':
    main()
