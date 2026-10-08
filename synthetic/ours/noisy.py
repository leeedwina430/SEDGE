import math
import torch
from torch import nn
from torch.nn import functional as F
from factorized_sedge.density import ConditionalGaussianMixture
from raw_diffusion_guidance.models import embedding

def context(u,z1,t):return torch.cat((u,z1,embedding(t,100,32)),1)
class NoisyGMM(nn.Module):
 def __init__(self):
  super().__init__();self.net=ConditionalGaussianMixture(35,1,8,128,3,.001)
 def log_prob(self,z,u,z1,t):return self.net.log_prob(z,context(u,z1,t))

class NoisyFlow(nn.Module):
 """Conditional RQS transform of standardized Z2; Gaussian base, identity spline tails.
 Learned conditional location/scale; 16 spline bins inside [-4,4]. No hard support.
 """
 def __init__(self,spline_fn,bins=16):
  super().__init__();self.spline_fn=spline_fn;self.bins=bins;self.B=4.
  self.net=nn.Sequential(nn.Linear(35,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3*bins+3))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
  with torch.no_grad():
   self.net[-1].bias[1]=math.log(math.expm1(1-.001));self.net[-1].bias[2+2*bins:]=math.log(math.expm1(1-.001))
 def log_prob(self,z,u,z1,t):
  return self._density(z,self.net(context(u,z1,t)).double())
 def _raw(self,params):
  raw=params[:,2:]
  endpoint=raw.new_full((len(raw),1),math.log(math.expm1(1-.001)))
  return torch.cat((raw[:,:2*self.bins],endpoint,raw[:,2*self.bins+1:-1],endpoint),1)
 def _density(self,z,params):
  loc=params[:,0];scale=F.softplus(params[:,1])+.001;raw=self._raw(params);v=(z.reshape(-1).double()-loc)/scale
  # Unit endpoint derivatives make the interior spline C1 with identity tails.
  endpoint=raw.new_full((len(raw),1),math.log(math.expm1(1-.001)))
  raw=torch.cat((raw[:,:2*self.bins],endpoint,raw[:,2*self.bins+1:-1],endpoint),1)
  y=((v+self.B)/(2*self.B)).clamp(0,1);mapped,logdet=self.spline_fn(y,raw,self.bins)
  inside=(v>=-self.B)&(v<=self.B);base=torch.where(inside,2*self.B*mapped-self.B,v);jac=torch.where(inside,logdet,torch.zeros_like(logdet))
  return -.5*base.square()-.5*math.log(2*math.pi)+jac-scale.log()

class CleanFlow(NoisyFlow):
 def __init__(self,spline_fn):
  super().__init__(spline_fn)
  self.net[0]=nn.Linear(1,128)
 def context_params(self,c):return self.net(c).double()
 def log_prob(self,z,c):
  return self._density(z,self.context_params(c))
 def sample(self,c,rng):
  params=self.context_params(c);loc=params[:,0];scale=F.softplus(params[:,1])+.001;raw=self._raw(params)
  v=torch.randn(len(c),device=c.device,dtype=torch.float64,generator=rng)
  mapped,_=self.spline_fn(((v+self.B)/(2*self.B)).clamp(0,1),raw,self.bins,inverse=True)
  value=torch.where(v.abs()<=self.B,2*self.B*mapped-self.B,v)
  return (loc+scale*value)[:,None].float()
