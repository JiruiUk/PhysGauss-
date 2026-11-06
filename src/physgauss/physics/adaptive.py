import torch


class AdaptiveCollocation:
    """EMA residual sampling with a uniform floor to retain domain coverage."""
    def __init__(self, count, device, uniform_fraction=0.3):
        self.scores = torch.ones(count, device=device)
        self.uniform_fraction = uniform_fraction

    def sample(self, count):
        p = self.scores.clamp_min(1e-6)
        p = (1-self.uniform_fraction)*p/p.sum() + self.uniform_fraction/len(p)
        return torch.multinomial(p, min(count, len(p)), replacement=False)

    def update(self, indices, residual):
        with torch.no_grad():
            score = residual.square().mean(-1).sqrt().clamp(1e-4, 1e4)
            self.scores[indices] = 0.9*self.scores[indices] + 0.1*score


def physics_scale(step, warmup, ramp):
    return min(1.0, max(0.0, (step - warmup) / ramp))
