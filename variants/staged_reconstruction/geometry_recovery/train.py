"""Geometry-repair entry; frozen prior code/state/results remain unchanged."""
import sys
import torch
from ..recovery import train as recovery
from ..prepare_data import read
from .model import GEOM_REPORT,GEOM_RUNS,BoundedScene


def main():
    if '--enable-geometry-recovery' not in sys.argv:raise SystemExit('Geometry repair is opt-in; pass --enable-geometry-recovery')
    sys.argv.remove('--enable-geometry-recovery');sys.argv.append('--enable-recovery')
    protocol=read(GEOM_REPORT/'S2_protocol.json');assert protocol['bounded_dynamic_scale']
    if '--preflight-only' not in sys.argv:
        from .freeze import verify
        verify()
    recovery.REC_REPORT,recovery.REC_RUNS=GEOM_REPORT,GEOM_RUNS;recovery.RecoveryScene=BoundedScene
    old_make=recovery.make_objective
    def make(original,configuration):
        previous=old_make(original,configuration)
        def objective(pred,gt):
            value=previous(pred,gt);r,g,b=gt
            # RGB premultiplied alpha; blue candidates are conditional appearance
            # evidence, with visible-yellow pixels excluded by the color test.
            static=(b>r*1.3+.02)&(b>g*1.05)
            if static.any():value=value+configuration['static_rgb_weight']*(pred[:,static]-gt[:,static]).abs().mean()
            return value
        return objective
    recovery.make_objective=make
    recovery.main()


if __name__=='__main__':main()
