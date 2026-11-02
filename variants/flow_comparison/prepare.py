"""Fork only the trainer/config hook; all original paths remain when flags off."""
from pathlib import Path
import shutil
from variants.monotonic_training.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json
HERE = Path(__file__).parent


def prepare():
    source = parent_prepare(); parent = runtime_hashes(source)
    target = HERE/'code'; receipt = HERE/'source_manifest.json'
    if target.exists():
        if read_json(receipt) != {'parent': parent, 'fork': runtime_hashes(target)}:
            raise ValueError('Flow fork changed')
        return target
    for relative in parent:
        p = target/relative; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, p)
    p = target/'scene_PIDG/gaussian_model.py'
    p.write_text(replace_once(p.read_text(encoding='utf8'), 'from variants.monotonic_training.model import configure_from_env',
        'from variants.flow_comparison.model import configure_from_env'), encoding='utf8')
    p = target/'train_pidg.py'; text = p.read_text(encoding='utf8')
    text = replace_once(text, 'from variants.monotonic_training.runtime import Tuning', 'from variants.flow_comparison.runtime import Tuning')
    text = replace_once(text, 'from variants.kinematic_resplit.checkpoint import restore_training_state, save_training_state',
        'from variants.flow_comparison.checkpoint import restore_training_state, save_training_state')
    text = replace_once(text, 'enable_flow_grad = True if opt.use_flow and',
        'enable_flow_grad = True if opt.use_flow and not tuning.observed_flow and')
    start = text.index('            cauchy_momentum_loss = cauchy_momentum_loss_sampled(')
    end = text.index('            loss = loss + opt.physics_loss_weight', start)
    old = text[start:end]
    text = text[:start]+('            if tuning.material_momentum:\n'
        '                cauchy_momentum_loss = tuning.momentum_loss(iteration, gaussians, deform, fid)\n'
        '            else:\n')+''.join('    '+line+'\n' for line in old.splitlines())+text[end:]
    text = replace_once(text, '            if opt.use_flow:\n',
        '            if opt.use_flow and tuning.observed_flow:\n'
        '                flow_loss = tuning.motion_loss(iteration, gaussians, deform, viewpoint_cam, deform_pkgs, render, pipe, background)\n'
        '                loss = loss + opt.flow_loss_weight * flow_loss\n'
        '            elif opt.use_flow:\n')
    # Full current warp dictionary uses the original trainer's local name.
    text = text.replace('viewpoint_cam, deform_pkgs, render,', 'viewpoint_cam, deform_pkgs_re, render,') if 'deform_pkgs_re = ' in text else text
    p.write_text(text, encoding='utf8')
    write_json(receipt, {'parent': parent, 'fork': runtime_hashes(target)})
    return target
