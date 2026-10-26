"""Cross-training-time coverage votes, bound to the full-state checkpoint."""
import torch


class CoverageAccumulator:
    def __init__(self, count, frames, device='cpu'):
        self.frames, self.device = frames, device
        self.reset(count)

    def reset(self, count):
        self.score = torch.zeros(count,device=self.device)
        self.votes = torch.zeros(count,self.frames,dtype=torch.bool,device=self.device)

    @torch.no_grad()
    def observe(self, frame, contributors, deficit):
        if not 0 <= frame < self.frames: raise ValueError('Non-training frame index')
        # Each footprint may occur at many deficit pixels; count its time once.
        ids = contributors.reshape(-1).long()
        values = deficit.expand_as(contributors).reshape(-1)
        valid = (ids >= 0) & (ids < len(self.score)) & (values > 0)
        ids, values = ids[valid], values[valid]
        if not ids.numel(): return
        self.score.scatter_add_(0,ids,values)
        self.votes[ids.unique(),frame] = True

    def select(self, min_views, limit):
        eligible = (self.votes.sum(1) >= min_views) & (self.score > 0)
        index = eligible.nonzero().flatten()
        if not index.numel() or limit <= 0: return index[:0]
        return index[self.score[index].topk(min(limit,len(index))).indices]

    def state_dict(self):
        return {'frames':self.frames,'score':self.score.cpu(),'votes':self.votes.cpu()}

    @classmethod
    def from_state(cls, state, device='cpu'):
        obj = cls(len(state['score']),state['frames'],device)
        obj.score, obj.votes = state['score'].to(device), state['votes'].to(device)
        if obj.votes.shape != (len(obj.score),obj.frames): raise ValueError('Invalid coverage window')
        return obj
