"""Frame-level hard-example sampling learned exclusively from training RGB/alpha."""
import math
import random


class FrameSampler:
    def __init__(self,names,probability=0.,fraction=.25):
        self.names=sorted(names)
        if not self.names or len(set(self.names))!=len(self.names):raise ValueError('Unique training frame names required')
        if not 0<=probability<=.5 or not 0<fraction<=.5:raise ValueError('Invalid bounded hard-frame sampling policy')
        self.probability=probability;self.fraction=fraction;self.ema={};self.counts={};self.replacements=0

    def observe(self,name,value):
        if name not in self.names:raise ValueError('Held-out frame may not enter the sampler')
        if not math.isfinite(value) or value<0:raise ValueError('Invalid training error')
        self.ema[name]=.9*self.ema.get(name,value)+.1*value;self.counts[name]=self.counts.get(name,0)+1

    def select(self,current,views,enabled=True):
        if not enabled or not self.probability:return current
        if any(self.counts.get(n,0)<2 for n in self.names):return current
        if random.random()>=self.probability:return current
        lookup={v.image_name:v for v in views}
        if set(lookup)!=set(self.names):raise ValueError('Sampler candidates differ from training membership')
        hard=sorted(self.names,key=lambda n:(-self.ema[n],n))[:max(1,math.ceil(self.fraction*len(self.names)))]
        self.replacements+=1;return lookup[random.choice(hard)]

    def state(self):return {'ema':self.ema,'counts':self.counts,'replacements':self.replacements}

    def restore(self,state):
        if not set(state['ema'])<=set(self.names) or not set(state['counts'])<=set(self.names):raise ValueError('Unexpected sampler frame IDs')
        if any(not math.isfinite(v) or v<0 for v in state['ema'].values()):raise ValueError('Invalid sampler state')
        self.ema=dict(state['ema']);self.counts=dict(state['counts']);self.replacements=state['replacements']
