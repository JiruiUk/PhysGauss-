"""Four-stage optimization with an exact neural-weight freeze audit."""
import bisect
import hashlib
import json


def neural_hash(gaussians,deform):
    digest=hashlib.sha256()
    for optimizer in (gaussians.optimizer,deform.optimizer):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class StagedOptimization:
    def __init__(self,config,run):
        self.config=config;self.run=run
        self.boundaries=config.get('stage_boundaries',[5000,10000,20000])
        if len(self.boundaries)!=3 or self.boundaries!=sorted(set(self.boundaries)) or self.boundaries[0]<1:
            raise ValueError('Three increasing positive stage boundaries required')
        self.base_g=None;self.base_d=None;self.last_stage=None;self.frozen_hash=None

    def apply(self,iteration,g,d,extension):
        stage='ABCD'[bisect.bisect_right(self.boundaries,iteration)]
        if self.base_g is None:
            self.base_g={p['name']:p['lr'] for p in g.optimizer.param_groups}
            self.base_d={p['name']:p['lr'] for p in d.optimizer.param_groups}
        for optimizer,base in ((g.optimizer,self.base_g),(d.optimizer,self.base_d)):
            for group in optimizer.param_groups:group['lr']=base[group['name']]
        g.update_learning_rate(max(0,iteration-1));d.update_learning_rate(max(0,iteration-1))
        multiplier=0. if stage=='C' else .1 if stage=='D' else 1.
        for optimizer in (g.optimizer,d.optimizer):
            for group in optimizer.param_groups:group['lr']*=multiplier
        extension.neural_step_enabled=stage!='C'
        extension.densification_enabled=stage!='C'
        extension.force_active=stage in ('C','D')
        if extension.material is not None:
            for parameter in extension.material.parameters():
                parameter.requires_grad_(self.config.get('learn_material_parameters',False) and stage in ('C','D'))
        if extension.force is not None:
            for parameter in extension.force.parameters():parameter.requires_grad_(stage in ('C','D'))
        for optimizer,base in ((extension.optimizer,self.config.get('material_lr',1e-5)),
                               (extension.force_optimizer,self.config.get('force_lr',1e-4))):
            if optimizer is not None:
                for group in optimizer.param_groups:group['lr']=base*(.1 if stage=='D' else 1. if stage=='C' else 0.)
        if stage!=self.last_stage:
            unchanged=None
            if self.last_stage=='C':
                unchanged=neural_hash(g,d)==self.frozen_hash
                if not unchanged:raise RuntimeError('Stage C neural weights changed despite requested freeze')
            if stage=='C':self.frozen_hash=neural_hash(g,d)
            event={'iteration':iteration,'stage':stage,'neural_lr_multiplier':multiplier,
                   'neural_step_enabled':extension.neural_step_enabled,'force_active':extension.force_active,
                   'stage_C_weights_unchanged':unchanged}
            with (self.run/'optimization_stages.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(event)+'\n')
            self.last_stage=stage

    def state(self):
        return {'base_g':self.base_g,'base_d':self.base_d,'last_stage':self.last_stage,'frozen_hash':self.frozen_hash}

    def restore(self,state):
        if state is None:raise ValueError('Missing staged-optimization checkpoint state')
        self.base_g=state['base_g'];self.base_d=state['base_d'];self.last_stage=state['last_stage'];self.frozen_hash=state['frozen_hash']
