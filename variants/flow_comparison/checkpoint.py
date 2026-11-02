"""Explicit branch permits only diagnostics relocation and enabling flow."""
from pathlib import Path
from variants.metric_tuning import checkpoint as metric
from variants.metric_tuning.run import read_json, sha256
save_training_state = metric.save_training_state


def validate_options(saved, current, receipt, digest):
    differences = {k for k in set(saved)|set(current) if saved.get(k) != current.get(k)}
    if not differences:
        return False
    if receipt is None or digest != receipt['identical_full_checkpoint_sha256']:
        raise ValueError('Options change requires identical, receipted branch checkpoint')
    if differences != {'save_image_dir', 'use_flow'} or saved['use_flow'] is not False or current['use_flow'] is not True:
        raise ValueError('Unapproved optimization change at branch')
    for options, side in ((saved, 'source'), (current, 'destination')):
        if Path(options['save_image_dir']).resolve() != (Path(receipt[side])/'diagnostics').resolve():
            raise ValueError('Unexpected diagnostics path')
    return True


def restore_training_state(path, gaussians, deform, opt, scene, tuning):
    state = metric.base.load_state(path)
    receipt_path = Path(path).parents[1]/'branch.json'
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    changed = validate_options(state['opt'], vars(opt), receipt, sha256(path))
    original = opt.save_image_dir, opt.use_flow
    try:
        if changed:
            opt.save_image_dir, opt.use_flow = state['opt']['save_image_dir'], state['opt']['use_flow']
        result = metric.base.restore_training_state(path, gaussians, deform, opt, scene)
    finally:
        opt.save_image_dir, opt.use_flow = original
    tuning.restore(path, result['iteration'])
    return result
