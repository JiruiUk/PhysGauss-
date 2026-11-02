"""Flag configuration preserves all original state and optimizer group layouts."""
import json
import os
from pathlib import Path
from variants.monotonic_training.model import configure_clock


def configure_from_env(net):
    path = os.environ.get('PIDG_METRIC_CONFIG')
    if not path:
        return net
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not config.get('enabled', False):
        return net
    configure_clock(net, config.get('velocity_clock_mode', 'legacy'))
    source = config.get('velocity_source', 'independent')
    if source not in ('independent', 'trajectory'):
        raise ValueError('Unknown velocity source')
    net.velocity_source = source
    if source == 'trajectory':
        net.velocity_head.requires_grad_(False)
    return net
