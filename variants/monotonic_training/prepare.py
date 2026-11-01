"""Independent H3 fork: same renderer/data/sampler; flag-controlled physical clock."""
from pathlib import Path
import shutil
from variants.integrity_tuning.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes,read_json,write_json
HERE=Path(__file__).parent


def prepare():
    source=parent_prepare();parent=runtime_hashes(source);target=HERE/'code';receipt=HERE/'source_manifest.json'
    if target.exists():
        if read_json(receipt)!={'parent':parent,'fork':runtime_hashes(target)}:raise ValueError('Monotonic fork changed')
        return target
    for relative in parent:
        p=target/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,p)
    p=target/'train_pidg.py'
    p.write_text(replace_once(p.read_text(encoding='utf8'),'from variants.integrity_tuning.runtime import Tuning',
                  'from variants.monotonic_training.runtime import Tuning'),encoding='utf8')
    p=target/'motion_utils/pidg.py';text=p.read_text(encoding='utf8')
    text=replace_once(text,'import torch\n','import torch\nfrom variants.monotonic_training.model import time_coordinate\n')
    text=replace_once(text,'        t_feat = self.time_embed(t)\n        t_normalized = torch.tanh(self.time_proj(t_feat)).squeeze(-1)',
                      '        t_normalized = time_coordinate(self, t)')
    p.write_text(text,encoding='utf8')
    p=target/'scene_PIDG/gaussian_model.py';text=p.read_text(encoding='utf8')
    text=replace_once(text,'        self.velocity_net = VelocityNet().to("cuda")',
        '        self.velocity_net = VelocityNet().to("cuda")\n        from variants.monotonic_training.model import configure_from_env\n        configure_from_env(self.velocity_net)')
    p.write_text(text,encoding='utf8')
    write_json(receipt,{'parent':parent,'fork':runtime_hashes(target)})
    return target
