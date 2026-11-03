"""Explicit repair wrapper around frozen S2 trainer; isolated runs/protocol/results."""
import sys
import numpy as np
import torch
from ..prepare_data import ROOT,REPORT,RUNS,read,write,sha
from ..stage2 import train as parent
from ..stage2.model import ObjectScene
from .prepare import REC_REPORT,REC_RUNS
from .coverage import coverage_loss

CONTEXT={}


class RecoveryScene(ObjectScene):
    def __init__(self,protocol):
        super().__init__(protocol)
        for obj,entry in zip(self.objects,read(REC_REPORT/'S2_data_receipt.json')['objects']):
            path=REC_RUNS/entry['file'];assert sha(path)==entry['sha256']
            with np.load(path) as data:
                with torch.no_grad():
                    obj._xyz.copy_(torch.tensor(data['xyz'],device='cuda'));obj.initial_xyz.copy_(obj._xyz)
                    obj.spacing=float(data['spacing']);obj._scaling.fill_(np.log(obj.spacing*.65))
                    self.centers[entry['object_id']].copy_(torch.tensor(data['center'],device='cuda'))

    def render(self,renderer,cam,t):
        package=super().render(renderer,cam,t)
        if torch.is_grad_enabled():
            CONTEXT.update({'positions':package['means3D_deform'][len(self.static.get_xyz):],
                'ids':self.object_ids,'camera':cam,'frame':int(round(float(t)*199))+1})
        return package


def make_objective(original,protocol):
    ledger=np.load(RUNS/'S2_observations.npz',allow_pickle=False)
    observations={int(n):(torch.tensor(c),o.tolist()) for n,c,o in zip(ledger['frame_ids'],ledger['components'],ledger['early_owner_to_component'])}
    count=0
    def objective(pred,gt):
        nonlocal count
        rgb=original(pred,gt);r,g,b=gt
        mask=(r>.25)&(g>.18)&(b<.8*torch.minimum(r,g))
        foreground=(pred[:,mask]-gt[:,mask]).abs().mean() if mask.any() else pred.sum()*0
        labels,mapping=observations[CONTEXT['frame']]
        cover=coverage_loss(CONTEXT['positions'],CONTEXT['ids'],CONTEXT['camera'],labels,mapping)
        loss=rgb+protocol['foreground_rgb_weight']*foreground+protocol['coverage_weight']*cover
        count+=1
        if count%20==0:
            import json
            with (REC_REPORT/'coverage_terms.jsonl').open('a') as stream:stream.write(json.dumps({'call':count,'frame':CONTEXT['frame'],
                'RGB':float(rgb.detach()),'foreground_RGB':float(foreground.detach()),'coverage':float(cover.detach()),
                'qualified_early_owner_to_component':mapping,'CMR_contact_off':True})+'\n')
        return loss
    return objective


def main():
    if '--enable-recovery' not in sys.argv:raise SystemExit('S2 repair is opt-in; pass --enable-recovery')
    sys.argv.remove('--enable-recovery')
    protocol=read(REC_REPORT/'S2_protocol.json');assert protocol['recovery_enabled'] and not protocol['cmr'] and not protocol['contact']
    from ..stage2.freeze import verify
    verify()
    if '--preflight-only' not in sys.argv:
        from .freeze import verify as verify_recovery
        verify_recovery()
    # Only this new entry point changes the parent module's runtime bindings.
    # Frozen source, prior attempt state, and original PIDG entry points stay unchanged.
    parent.REPORT=REC_REPORT;parent.RUNS=REC_RUNS;parent.ObjectScene=RecoveryScene
    parent.objective=make_objective(parent.objective,protocol)
    old_evaluate=parent.evaluate
    def evaluate(renderer,scene,step,split='val',width=1600,save_images=False):
        # Parent numeric evaluation is reused verbatim. Separate review exports
        # the recovery images; do not overwrite the original attempt's images.
        return old_evaluate(renderer,scene,step,split,width,False)
    parent.evaluate=evaluate
    parent.main()


if __name__=='__main__':main()
