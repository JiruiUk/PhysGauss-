import math
import torch
from ..recovery import train as recovery
from ..recovery.train import RecoveryScene
from ..prepare_data import REPORT,RUNS

GEOM_REPORT=REPORT/'S2_geometry_recovery';GEOM_RUNS=RUNS/'S2_geometry_recovery'


def bounded_scale(canonical_log_scale,delta,spacing,enabled=False):
    if not enabled:return canonical_log_scale.exp()+delta
    # At canonical=.65*spacing and delta=0, output remains .65*spacing.
    # The strictly positive bounds prevent additive cancellation of a group.
    shift=math.log((.65-.15)/(2.5-.65))-math.log(.65)
    return spacing*(.15+2.35*torch.sigmoid(canonical_log_scale-math.log(spacing)+delta+shift))


class BoundedScene(RecoveryScene):
    def __init__(self,protocol):
        old_report,old_runs=recovery.REC_REPORT,recovery.REC_RUNS
        recovery.REC_REPORT,recovery.REC_RUNS=GEOM_REPORT,GEOM_RUNS
        try:super().__init__(protocol)
        finally:recovery.REC_REPORT,recovery.REC_RUNS=old_report,old_runs
        self.bounded_dynamic_scale=bool(protocol.get('bounded_dynamic_scale',False))

    def object_at(self,k,t):
        result=super().object_at(k,t)
        if not self.bounded_dynamic_scale:return result
        obj=self.objects[k]
        state=self.deforms[k].step(obj.get_xyz,obj.get_xyz.new_full((len(obj.get_xyz),1),float(t)))
        result.get_scaling=bounded_scale(obj._scaling,state['d_scaling'],obj.spacing,True)
        return result
