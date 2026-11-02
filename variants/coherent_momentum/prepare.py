"""Copy the frozen extended architecture; change only trainer runtime import."""
from pathlib import Path
import shutil
from variants.momentum_guard.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json
HERE = Path(__file__).parent


def prepare():
    source = parent_prepare(); parent = runtime_hashes(source)
    target = HERE/'code'; receipt = HERE/'source_manifest.json'
    if target.exists():
        if read_json(receipt) != {'parent': parent, 'fork': runtime_hashes(target)}:
            raise ValueError('Momentum guard fork changed')
        return target
    for relative in parent:
        p = target/relative; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, p)
    p = target/'train_pidg.py'
    text = replace_once(p.read_text(encoding='utf8'),
        'from variants.momentum_guard.runtime import Tuning', 'from variants.coherent_momentum.runtime import Tuning')
    p.write_text(text, encoding='utf8')
    write_json(receipt, {'parent': parent, 'fork': runtime_hashes(target)})
    return target
