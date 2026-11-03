"""S3 entry fixing graph-cache recovery through a new frozen adapter."""
import sys
import torch
from pathlib import Path
from ..prepare_data import ROOT,REPORT,RUNS,read,write,sha
from ..stage3 import train as base
from .runtime import save_complete,restore_graphs
from .physics import CurrentWorldField

OriginalScene=base.PhysicsScene
original_freeze=base.freeze


class CompleteScene(OriginalScene):
    def __init__(self,parent_state):
        super().__init__(parent_state)
        self.physics=CurrentWorldField(len(self.objects))

    def load_state_dict(self,state,*args,**kwargs):
        result=super().load_state_dict(state,*args,**kwargs)
        if hasattr(self,'physics') and any(k.startswith('physics.') for k in state):
            directory=RUNS/'S3'/('_smoke' if '--smoke' in sys.argv else 'formal')
            checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
            if 'physics_graphs' not in checkpoint:raise RuntimeError('Old incomplete graph-cache checkpoint cannot be relabeled as full recovery')
            restore_graphs(self,checkpoint['physics_graphs'])
        return result


def freeze(protocol):
    original_freeze(protocol)
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'tests/test_staged_graph_state.py']
    values={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths}
    receipt=REPORT/'S3/state_recovery_source_freeze.json'
    if receipt.exists():assert read(receipt)==values
    else:write(receipt,values)
    policy=REPORT/'S3/spatial_operator_contract.json'
    value={'correction':'Use actual current-world same-bank kNN, instead of canonical neighbors after nonaffine deformation',
        'support_schedule':[64,128,256],'minimum_valid_fraction_unchanged':.25,
        'rank_ratio_unchanged':1e-6,'invalid_neighborhoods':'Fail, never zero residual',
        'cache':'Current-world neighbors recomputed each objective call; inherited empty caches explicitly checkpointed',
        'no_stress_geometry_or_physical_ID_GT':True}
    if policy.exists():assert read(policy)==value
    else:write(policy,value)


def main():
    base.PhysicsScene=CompleteScene;base.save=save_complete;base.freeze=freeze
    base.main()


if __name__=='__main__':main()
