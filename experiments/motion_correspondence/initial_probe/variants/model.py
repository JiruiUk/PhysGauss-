"""History-based multi-hypothesis association with explicit abstention.

States are image positions/velocities, not PIDG physical velocity predictions.
Costs and margins are conditional hypothesis scores, NOT calibrated physical
identity probabilities. Merged image components are never split by proximity.
"""
from dataclasses import dataclass,asdict
from itertools import product
import copy
import numpy as np


@dataclass(frozen=True)
class Config:
    use_history: bool=True
    uncertainty_guard: bool=True
    measurement_sigma: float=.75
    acceleration_sigma: float=5000.
    initial_velocity_sigma: float=1000.
    interaction_velocity_sigma: float=800.
    mahalanobis_gate: float=9.21
    assignment_margin: float=2.3
    beam_width: int=32
    max_missing_frames: int=30
    merged_area_ratio: float=1.5


@dataclass
class Observation:
    xy: np.ndarray
    area: float
    component: int
    mask: np.ndarray|None=None


@dataclass
class State:
    mean: np.ndarray
    covariance: np.ndarray
    area: float
    last_seen: float
    hidden: bool=False


def predict(state,dt,config):
    if dt<0:raise ValueError('Time must increase')
    f=np.eye(4);f[:2,2:]=np.eye(2)*dt
    mean=state.mean.copy()
    if config.use_history:mean=f@mean
    g=np.vstack([np.eye(2)*dt*dt/2,np.eye(2)*dt])
    covariance=f@state.covariance@f.T+g@g.T*config.acceleration_sigma**2
    return State(mean,(covariance+covariance.T)/2,state.area,state.last_seen,state.hidden)


def innovation(state,observation,config):
    s=state.covariance[:2,:2]+np.eye(2)*config.measurement_sigma**2
    delta=np.asarray(observation.xy)-state.mean[:2]
    d2=float(delta@np.linalg.solve(s,delta))
    # Gaussian match likelihood relative to uniform image clutter, including
    # covariance volume so broad predictions do not become confident matches.
    cost=.5*d2+.5*np.linalg.slogdet(s)[1]+np.log(2*np.pi)-np.log(400*225)
    cost+=.5*(np.log(max(observation.area,1)/max(state.area,1))/.5)**2
    return cost,d2


def update(state,observation,time,config):
    r=np.eye(2)*config.measurement_sigma**2
    h=np.zeros((2,4));h[:,:2]=np.eye(2)
    s=h@state.covariance@h.T+r;k=np.linalg.solve(s,h@state.covariance).T
    mean=state.mean+k@(observation.xy-h@state.mean)
    # Joseph form maintains finite symmetric positive covariance.
    a=np.eye(4)-k@h;p=a@state.covariance@a.T+k@r@k.T
    return State(mean,(p+p.T)/2,.8*state.area+.2*observation.area,time,False)


def is_merged(observation,states,config):
    if observation.mask is None:return False
    h,w=observation.mask.shape;contains=0;areas=[]
    for s in states:
        x,y=np.rint(s.mean[:2]).astype(int)
        inside=0<=x<w and 0<=y<h and observation.mask[y,x]
        contains+=int(inside)
        if inside:areas.append(s.area)
    return contains>=2 or (contains>=1 and observation.area>config.merged_area_ratio*np.median(areas))


class MotionTracker:
    def __init__(self,observations,time,config=Config()):
        if not observations:raise ValueError('Need an initially separated visible seed set')
        self.config=config;self.time=float(time);self.count=len(observations)
        p=np.diag([config.measurement_sigma**2]*2+[config.initial_velocity_sigma**2]*2)
        states=[State(np.r_[o.xy,[0.,0.]],p.copy(),o.area,time) for o in observations]
        self.hypotheses=[{'states':states,'cost':0.,'assignment':tuple(range(self.count))}]
        self.last_result=None

    def state_dict(self):
        return {'config':asdict(self.config),'time':self.time,'count':self.count,'hypotheses':[
            {'cost':h['cost'],'assignment':list(h['assignment']),'states':[
                {'mean':s.mean.tolist(),'covariance':s.covariance.tolist(),'area':s.area,
                 'last_seen':s.last_seen,'hidden':s.hidden} for s in h['states']]} for h in self.hypotheses]}

    @classmethod
    def from_state_dict(cls,value):
        self=cls.__new__(cls);self.config=Config(**value['config']);self.time=value['time'];self.count=value['count'];self.last_result=None
        self.hypotheses=[{'cost':h['cost'],'assignment':tuple(h['assignment']),'states':[
            State(np.array(s['mean']),np.array(s['covariance']),s['area'],s['last_seen'],s['hidden']) for s in h['states']]} for h in value['hypotheses']]
        return self

    def step(self,observations,time):
        if time<=self.time:raise ValueError('Time must strictly increase')
        c=self.config;dt=time-self.time;candidates=[]
        for hypothesis in self.hypotheses:
            states=[predict(s,dt,c) for s in hypothesis['states']]
            merged={j for j,o in enumerate(observations) if is_merged(o,states,c)}
            choices=[];costs=[]
            for s in states:
                options=[-1];values={-1:0.}
                expired=(time-s.last_seen)*199>c.max_missing_frames
                if not expired:
                    for j,o in enumerate(observations):
                        if j in merged:continue
                        score,d2=innovation(s,o,c)
                        if d2<=c.mahalanobis_gate:options.append(j);values[j]=score
                choices.append(options);costs.append(values)
            updates=[]
            for i,s in enumerate(states):
                values={}
                for j in choices[i]:
                    if j>=0:values[j]=update(s,observations[j],time,c)
                    else:
                        hidden=copy.deepcopy(s)
                        if c.uncertainty_guard and not hidden.hidden:
                            hidden.covariance[2:,2:]+=np.eye(2)*c.interaction_velocity_sigma**2
                        hidden.hidden=True;values[j]=hidden
                updates.append(values)
            forecast=[s.mean[:2].tolist() for s in states]
            forecast_covariance=[s.covariance[:2,:2].tolist() for s in states]
            for assignment in product(*choices):
                matched=[j for j in assignment if j>=0]
                if len(set(matched))!=len(matched):continue
                # Predicted/updated states are immutable after construction;
                # sharing cached per-association updates avoids repeated solves.
                new=[updates[i][j] for i,j in enumerate(assignment)]
                candidates.append({'states':new,'cost':hypothesis['cost']+sum(costs[i][j] for i,j in enumerate(assignment)),
                    'assignment':tuple(assignment),'merged':sorted(merged),'forecast':forecast,
                    'forecast_covariance':forecast_covariance})
        candidates.sort(key=lambda h:h['cost'])
        best=candidates[0];base=best['cost'];accepted=[];margins=[];alternatives=[]
        for i,j in enumerate(best['assignment']):
            alt=min((h['cost'] for h in candidates if h['assignment'][i]!=j),default=np.inf)
            margin=alt-base
            # No alternative means the only possible value is missing, or a
            # gate removed all rivals; do not manufacture infinite confidence.
            accepted.append(j if j>=0 and np.isfinite(margin) and margin>=c.assignment_margin else -1)
            margins.append(float(margin) if np.isfinite(margin) else None)
            grouped={}
            for h in candidates:
                k=h['assignment'][i];grouped[k]=min(grouped.get(k,np.inf),h['cost']-base)
            alternatives.append([{'observation':k,'relative_cost':float(v)} for k,v in sorted(grouped.items(),key=lambda x:x[1])[:5]])
        self.hypotheses=candidates[:c.beam_width]
        for h in self.hypotheses:h['cost']-=base
        self.time=float(time)
        result={'time':time,'best_assignment':list(best['assignment']),'accepted_assignment':accepted,'assignment_margins':margins,
          'alternatives':alternatives,'forecast':best['forecast'],'forecast_covariance':best['forecast_covariance'],
          'merged_components':best['merged'],'retained_hypotheses':len(self.hypotheses),'generated_hypotheses':len(candidates),
          'scores_are_not_calibrated_probabilities':True}
        self.last_result=result;return result


def optional_visible_propagation(labels,components,forward,backward,*,enabled=False,tracker=None,time=None,observations=None):
    """Flags off delegates exactly to the prior read-only visible propagator.

    New outputs are candidate IMAGE identities only, never Gaussian/material IDs.
    """
    if not enabled:
        from output.surface_support.core import propagate
        return propagate(labels,components,forward,backward)
    if tracker is None or time is None or observations is None:raise ValueError('Missing motion state')
    result=tracker.step(observations,time)
    output=np.full(components.shape,-1,dtype=np.int64)
    for identity,j in enumerate(result['accepted_assignment']):
        if j>=0:output[components==observations[j].component]=identity
    return output,result
