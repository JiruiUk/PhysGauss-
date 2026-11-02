"""Independent fork; original trainers and all previous hash locks remain intact."""
from pathlib import Path
import shutil
from variants.flow_comparison.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes, read_json, write_json
HERE = Path(__file__).parent


def prepare():
    source = parent_prepare(); parent = runtime_hashes(source)
    target = HERE/'code'; receipt = HERE/'source_manifest.json'
    if target.exists():
        if read_json(receipt) != {'parent': parent, 'fork': runtime_hashes(target)}:
            raise ValueError('Unified fork changed')
        return target
    for relative in parent:
        p = target/relative; p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, p)
    for relative in ['scene_PIDG/gaussian_model.py', 'train_pidg.py']:
        p = target/relative
        p.write_text(p.read_text(encoding='utf-8').replace('variants.flow_comparison.', 'variants.unified_flow.'), encoding='utf-8')
    p = target/'scene_PIDG/network.py'; text = p.read_text(encoding='utf-8')
    text = replace_once(text, '            metric_residual_width=0,\n', '            metric_residual_width=0,\n            unified_stress_head=False,\n')
    text = replace_once(text, '        self.attention_score = None\n',
        '        self.attention_score = None\n'
        '        from variants.unified_flow.model import stress_head, configure_deform\n'
        '        self.shared_stress = stress_head(width) if unified_stress_head else None\n'
        '        configure_deform(self)\n')
    p.write_text(text, encoding='utf-8')
    p = target/'scene_PIDG/deform_model.py'; text = p.read_text(encoding='utf-8')
    start = text.index('        l = [\n'); end = text.index('        self.optimizer = ', start)
    text = text[:start]+'        from variants.unified_flow.model import optimizer_groups\n        l = optimizer_groups(self, training_args)\n\n'+text[end:]
    p.write_text(text, encoding='utf-8')
    write_json(receipt, {'parent': parent, 'fork': runtime_hashes(target)})
    return target
