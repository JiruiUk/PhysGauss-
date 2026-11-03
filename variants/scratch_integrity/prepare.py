"""Only replace the C1 Tuning import; original trainer and renderer are retained."""
from pathlib import Path
import shutil
from variants.color_repair.prepare import prepare as parent_prepare
from variants.metric_tuning.prepare import replace_once
from variants.metric_tuning.run import runtime_hashes,read_json,write_json
HERE=Path(__file__).parent


def prepare():
    source=parent_prepare();parent=runtime_hashes(source);target=HERE/'code';receipt=HERE/'source_manifest.json'
    if target.exists():
        assert read_json(receipt)=={'parent':parent,'fork':runtime_hashes(target)}
        return target
    for name in parent:
        p=target/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,p)
    p=target/'train_pidg.py';p.write_text(replace_once(p.read_text(encoding='utf-8'),
        'from variants.color_repair.runtime import Tuning','from variants.scratch_integrity.runtime import Tuning'),encoding='utf-8')
    assert runtime_hashes(source)==parent
    write_json(receipt,{'parent':parent,'fork':runtime_hashes(target)})
    return target
