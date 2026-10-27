"""Training-only color moments for conservative coverage clones; no learned state."""
import torch


class ColorEvidence:
    def __init__(self,count,device='cpu'):
        self.mass=torch.zeros(count,device=device)
        self.first=torch.zeros(count,3,device=device)
        self.second=torch.zeros(count,3,device=device)

    @torch.no_grad()
    def observe(self,contributors,deficit,rgb):
        ids=contributors.reshape(-1).long();pixels=deficit.numel()
        weights=deficit.reshape(1,-1).expand(contributors.shape[0],-1).reshape(-1)
        valid=(ids>=0)&(ids<len(self.mass))&(weights>0)
        positions=valid.nonzero().flatten()
        if not len(positions):return
        ids=ids[positions];weights=weights[positions]
        colors=rgb.permute(1,2,0).reshape(-1,3)[positions.remainder(pixels)]
        self.mass.scatter_add_(0,ids,weights)
        self.first.index_add_(0,ids,colors*weights[:,None])
        self.second.index_add_(0,ids,colors.square()*weights[:,None])

    def mean_std(self,ids):
        denominator=self.mass[ids].clamp_min(1e-8)[:,None]
        mean=self.first[ids]/denominator
        variance=(self.second[ids]/denominator-mean.square()).clamp_min(0.)
        return mean,variance.sqrt()

    def state_dict(self):
        return {k:getattr(self,k).cpu() for k in ('mass','first','second')}

    @classmethod
    def from_state(cls,state,device):
        result=cls(len(state['mass']),device)
        for key in ('mass','first','second'):setattr(result,key,state[key].to(device))
        if result.first.shape!=(len(result.mass),3) or result.second.shape!=result.first.shape:
            raise ValueError('Color-evidence dimensions changed')
        return result
