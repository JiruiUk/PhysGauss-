"""One bounded shared Young modulus; no per-Gaussian physical parameters."""
import math
import torch
from torch import nn
from pidg_ext.losses import constitutive_consistency


class SharedYoung(nn.Module):
    def __init__(self, young=1., poisson=.3, learn=False, lower=.05, upper=20.):
        super().__init__()
        if not all(math.isfinite(v) for v in (young,poisson,lower,upper)) or \
           not (0<lower<young<upper) or not (-1<poisson<.5):
            raise ValueError('Invalid effective stiffness, bounds, or fixed Poisson ratio')
        self.initial,self.poisson,self.lower,self.upper=young,poisson,lower,upper
        ratio=torch.tensor((young-lower)/(upper-lower),dtype=torch.float64)
        self.raw_E=nn.Parameter(torch.logit(ratio),requires_grad=learn)

    @property
    def young(self):
        return self.lower+(self.upper-self.lower)*self.raw_E.sigmoid()

    def parameters_for(self,young):
        return {'mu':young/(2*(1+self.poisson)), 'kappa':young/(3*(1-2*self.poisson))}

    def forward(self):
        return self.parameters_for(self.young)

    def prior(self):
        return (self.young/self.initial).log().square()


@torch.no_grad()
def conditional_profile(pred,F,material,confidence,valid):
    points=[]
    for factor in (.1,.3,1.,3.,10.):
        value=material.initial*factor
        raw,_=constitutive_consistency(pred.detach(),F.detach(),material.parameters_for(value),confidence,valid)
        points.append({'E':value,'loss':raw.item()})
    return {'conditional_only':True,'data_identifiability':'not_established',
            'reason':'Neural stress, unknown forces, and normalized physical scales remain confounded',
            'points':points}
