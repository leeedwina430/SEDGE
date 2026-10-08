"""Raw unconditional, conditional, and noisy-density networks."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from factorized_sedge.networks import ResidualMLP
from .common import RECIPE,WIDTHS,COUNTS,keyed


def embedding(t,steps,dim=64):
    half=dim//2
    frequencies=torch.exp(-math.log(10000)*torch.arange(half,device=t.device,dtype=torch.float32)/max(half-1,1))
    angles=(t.float()/max(steps-1,1))[:,None]*frequencies[None,:]
    return torch.cat((torch.sin(angles),torch.cos(angles)),dim=-1)

def condition(z,present):
    if not torch.is_tensor(present):present=z.new_full((len(z),1),float(present))
    return torch.cat((z*present,present),-1)

class Schedule(nn.Module):
    def __init__(self,recipe):
        super().__init__();self.recipe=dict(recipe);self.steps=recipe['timesteps']
        b=torch.linspace(recipe['beta_start'],recipe['beta_end'],self.steps);a=1-b;ab=torch.cumprod(a,0)
        previous=torch.cat((torch.ones(1),ab[:-1]))
        for name,value in [('betas',b),('alphas',a),('alpha_bars',ab),('posterior_variance',b*(1-previous)/(1-ab))]:self.register_buffer(name,value)
    def corrupt(self,x,t,noise):
        a=self.alpha_bars[t,None];return a.sqrt()*x+(1-a).sqrt()*noise

class DDPM(Schedule):
    def __init__(self,d,k,kind,width,recipe):
        super().__init__(recipe);self.d=d;self.k=k;self.kind=kind
        context=0 if kind=='prior' else k+(kind=='cfg')
        self.denoiser=ResidualMLP(d+context+recipe['time_dim'],d,width,recipe['depth'],0.)
    def predict_noise(self,x,t,z=None,present=True):
        parts=[x]
        if self.kind!='prior':
            assert z is not None
            parts.append(condition(z,present) if self.kind=='cfg' else z)
        parts.append(embedding(t,self.steps,self.recipe['time_dim']))
        return self.denoiser(torch.cat(parts,-1))
    def training_loss(self,x,z,generator):
        t=torch.randint(self.steps,(len(x),),device=x.device,generator=generator)
        n=torch.randn(x.shape,device=x.device,generator=generator,dtype=x.dtype)
        m=(torch.rand((len(x),1),device=x.device,generator=generator)>=self.recipe['condition_dropout']).to(x.dtype) if self.kind=='cfg' else True
        return F.mse_loss(self.predict_noise(self.corrupt(x,t,n),t,z,m),n)

class Forward(nn.Module):
    def __init__(self,d,k,width):
        super().__init__();self.d=d;self.k=k
        self.net=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,k))
    def forward(self,x):return self.net(x)
    def training_loss(self,x,z,generator):return F.mse_loss(self(x),z)

class NoisyDensity(Schedule):
    def __init__(self,d,k,width,recipe):
        super().__init__(recipe);self.d=d;self.k=k;self.components=recipe['density_components']
        o=self.components*(1+k+k*(k+1)//2)
        self.net=nn.Sequential(nn.Linear(d+recipe['time_dim'],width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,o))
        tril=torch.tril_indices(k,k);self.register_buffer('tril_rows',tril[0]);self.register_buffer('tril_cols',tril[1])
    def distribution_parameters(self,x,t):
        raw=self.net(torch.cat((x,embedding(t,self.steps,self.recipe['time_dim'])),-1)).reshape(len(x),self.components,-1)
        weights=F.log_softmax(raw[:,:,0],-1);means=raw[:,:,1:1+self.k];values=raw[:,:,1+self.k:]
        diagonal=self.tril_rows==self.tril_cols
        values=torch.where(diagonal,F.softplus(values)+self.recipe['density_cholesky_floor'],values)
        lower=x.new_zeros((len(x),self.components,self.k,self.k));lower[:,:,self.tril_rows,self.tril_cols]=values
        covariance=lower@lower.transpose(-1,-2)+self.recipe['density_covariance_floor']*torch.eye(self.k,device=x.device,dtype=x.dtype)
        return weights,means,torch.linalg.cholesky(covariance)
    def log_prob(self,z,x,t):
        w,m,l=self.distribution_parameters(x,t)
        residual=torch.linalg.solve_triangular(l,(z[:,None,:]-m).unsqueeze(-1),upper=False).squeeze(-1)
        logp=-.5*(self.k*math.log(2*math.pi)+residual.square().sum(-1))-torch.diagonal(l,dim1=-2,dim2=-1).log().sum(-1)
        return torch.logsumexp(w+logp,-1)
    def training_loss(self,x,z,generator):
        t=torch.randint(self.steps,(len(x),),device=x.device,generator=generator)
        n=torch.randn(x.shape,device=x.device,generator=generator,dtype=x.dtype)
        return -self.log_prob(z,self.corrupt(x,t,n),t).mean()

def make_model(cohort,seed,kind,recipe=None,width=None,device='cpu'):
    recipe=recipe or RECIPE;d=9 if cohort=='cov_all' else 3;k=4 if d==9 else 2
    h=width or WIDTHS[d][kind]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(keyed('raw-guidance-v1',cohort,seed,kind,'initialization'))
        if kind in ('prior','cfg','direct'):model=DDPM(d,k,kind,h,recipe)
        elif kind=='forward':model=Forward(d,k,h)
        elif kind=='density':model=NoisyDensity(d,k,h,recipe)
        else:raise ValueError(kind)
    if width is None:assert sum(p.numel() for p in model.parameters())==COUNTS[d][kind]
    return model.to(device)
