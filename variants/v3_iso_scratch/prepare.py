"""Capture identical zero-step state without loading or updating any model weights."""
from pathlib import Path
import shutil
from variants.v3_isotropic.prepare import prepare as prepare_parent
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json

HERE = Path(__file__).parent
TARGET = HERE/'code'
CAPTURE_HOOK = '''    if tuning.enabled and tuning.config.get("allow_initial_capture", False) and os.environ.get("PIDG_CAPTURE_INITIAL_ONLY") == "1":
        if resume_checkpoint:
            raise ValueError("Initial state capture must not restore a checkpoint")
        iteration = 0
        save_training_state(locals(), os.path.join(args.model_path, "checkpoints", "resume.pt"))
        print("INITIAL_STATE_CAPTURED iteration=0 optimizer_updates=0", flush=True)
        raise SystemExit(75)
'''


def prepare():
    source = prepare_parent()
    original = runtime_hashes(source)
    receipt = HERE/'source_manifest.json'
    if TARGET.exists():
        if read_json(receipt) != {'parent': original, 'fork': runtime_hashes(TARGET)}:
            raise ValueError('Scratch source changed')
        return TARGET
    for relative in original:
        path = TARGET/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, path)
    path = TARGET/'train_pidg.py'
    text = replace_once(path.read_text(encoding='utf-8'),
        'from variants.v3_isotropic.runtime import Tuning', 'from variants.v3_iso_scratch.runtime import Tuning')
    anchor = '    progress_bar = tqdm(total=opt.iterations, initial=first_iteration-1, desc="Training progress")\n'
    text = replace_once(text, anchor, CAPTURE_HOOK+anchor)
    path.write_text(text, encoding='utf-8')
    assert runtime_hashes(source) == original
    write_json(receipt, {'parent': original, 'fork': runtime_hashes(TARGET)})
    return TARGET
