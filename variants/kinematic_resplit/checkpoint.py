"""Exact full-state branch with an explicitly receipted diagnostics-path move."""
from pathlib import Path
from variants.metric_tuning import checkpoint as metric
from variants.metric_tuning.run import read_json, sha256

save_training_state = metric.save_training_state


def validate_relocation(saved, current, receipt, digest):
    differences = {k for k in set(saved) | set(current) if saved.get(k) != current.get(k)}
    if not differences: return False
    if differences != {'save_image_dir'} or receipt is None:
        raise ValueError('Training options differ beyond a receipted output relocation')
    if digest != receipt['identical_full_checkpoint_sha256']:
        raise ValueError('Relocated checkpoint is not the identical branch state')
    if Path(saved['save_image_dir']).resolve() != (Path(receipt['source'])/'diagnostics').resolve():
        raise ValueError('Unexpected source diagnostics path')
    if Path(current['save_image_dir']).resolve() != (Path(receipt['destination'])/'diagnostics').resolve():
        raise ValueError('Unexpected destination diagnostics path')
    return True


def restore_training_state(path, gaussians, deform, opt, scene, tuning):
    state = metric.base.load_state(path)
    receipt_path = Path(path).parents[1]/'branch.json'
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    relocated = validate_relocation(state['opt'], vars(opt), receipt, sha256(path))
    original_path = opt.save_image_dir
    try:
        if relocated: opt.save_image_dir = state['opt']['save_image_dir']
        # The original restore checks every remaining optimization parameter and
        # restores tensors, optimizers, camera stack and RNG without alteration.
        result = metric.base.restore_training_state(path, gaussians, deform, opt, scene)
    finally:
        opt.save_image_dir = original_path
    tuning.restore(path, result['iteration'])
    return result
