"""Flow/Bernoulli densities; no GMM training. Shared existing spline code."""
import importlib.util,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fm=load('bench_spline',ROOT.parent/'linear_bounded_flow/model.py')
nm=load('bench_noisy',ROOT.parent/'linear_six_guidance/model.py')

def time_context(x,t):
    w=torch.exp(-math.log(10000)*torch.arange(8,device=x.device)/8)
    a=t[:,None].float()*w[None,:]
    return torch.cat([x,a.sin(),a.cos()],1)

class Density(nn.Module):
    def __init__(self,context_dim,outdim=1,binary=False,joint=False):
        super().__init__();self.binary=binary;self.joint=joint;self.outdim=outdim;self.parts=nn.ModuleList()
        for j in range(outdim):
            d=context_dim+(j if joint else 0)
            if binary:self.parts.append(nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1)))
            else:
                f=nm.CleanFlow(fm.spline);f.net[0]=nn.Linear(d,128);self.parts.append(f)
    def log_prob(self,z,c):
        terms=[]
        for j,model in enumerate(self.parts):
            ctx=torch.cat([c,z[:,:j]],1) if self.joint and j else c
            if self.binary:terms.append(-F.binary_cross_entropy_with_logits(model(ctx).flatten(),z[:,j],reduction='none'))
            else:terms.append(model.log_prob(z[:,j:j+1],ctx))
        return torch.stack(terms,1).sum(1)
    @torch.no_grad()
    def sample(self,c):
        out=[]
        for j,model in enumerate(self.parts):
            ctx=torch.cat([c,*out],1) if self.joint and j else c
            v=torch.bernoulli(model(ctx).sigmoid()) if self.binary else model.sample(ctx,None)
            out.append(v)
        return torch.cat(out,1)

def features(c):
    cols=[torch.ones(len(c),device=c.device,dtype=c.dtype)]
    cols.extend(c[:,j]**p for p in [1,2,3] for j in range(c.shape[1]))
    if c.shape[1]>1:cols.append(c[:,0]*c[:,1])
    return torch.stack(cols,1)

class BoundedFlow(nn.Module):
    """Spline density on training-fitted polynomial lower/upper envelopes."""
    def __init__(self,c,z):
        super().__init__();self.bins=16
        phi=features(c.double()).numpy();zz=z.double().flatten().numpy();n,k=phi.shape
        # Joint LP: lower <= observations <= upper; minimize mean interval width.
        A=np.concatenate([np.column_stack([phi,np.zeros_like(phi)]),np.column_stack([np.zeros_like(phi),-phi]),np.column_stack([phi,-phi])])
        y=np.r_[zz,-zz,np.full(n,-1e-4)]
        sol=linprog(np.r_[-phi.mean(0),phi.mean(0)],A_ub=A,b_ub=y,bounds=[(None,None)]*(2*k),method='highs');assert sol.success
        self.register_buffer('lower',torch.tensor(sol.x[:k]));self.register_buffer('upper',torch.tensor(sol.x[k:]));self.lower[0]-=1e-4;self.upper[0]+=1e-4
        self.net=nn.Sequential(nn.Linear(c.shape[1],128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,49))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():self.net[-1].bias[32:]=math.log(math.expm1(1-.001))
    def interval(self,c):
        p=features(c.double());return p@self.lower,p@self.upper
    def log_prob(self,z,c):
        a,b=self.interval(c);width=(b-a).clamp_min(1e-10);y=(z.flatten().double()-a)/width
        _,ld=fm.spline(y.clamp(0,1),self.net(c),self.bins)
        return torch.where((b>a)&(y>=0)&(y<=1),ld-width.log(),-torch.inf)
    @torch.no_grad()
    def sample(self,c):
        a,b=self.interval(c);u=torch.rand(len(c),device=c.device,dtype=torch.float64);v,_=fm.spline(u,self.net(c),self.bins,inverse=True)
        return (a+(b-a)*v).float()[:,None]

def fit_density(model,x,z,vx,vz,path,steps,seed,corrupt=None):
    path.mkdir(parents=True,exist_ok=True)
    if (path/'best.pt').exists():model.load_state_dict(torch.load(path/'best.pt',weights_only=False)['model']);return model.eval().requires_grad_(False)
    torch.manual_seed(seed);model=model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0);best=math.inf;history=[]
    def batch(xx,zz):return corrupt(xx,zz) if corrupt else (xx,zz)
    with torch.no_grad():vc,vt=batch(vx.cuda(),vz.cuda())
    for step in range(1,steps+1):
        ix=torch.randint(len(x),(min(512,len(x)),));c,t=batch(x[ix].cuda(),z[ix].cuda());lp=model.log_prob(t,c);ok=torch.isfinite(lp)
        if not ok.any():raise RuntimeError('Density training has no supported points')
        loss=-lp[ok].mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad():val=torch.cat([model.log_prob(vt[j:j+512],vc[j:j+512]) for j in range(0,len(vc),512)]);finite=torch.isfinite(val);score=float(-val[finite].mean()) if finite.any() else math.inf
            history.append(dict(step=step,supported_nll=score,outside_support=int((~finite).sum())))
            if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};beststep=step
    if not math.isfinite(best):raise RuntimeError('No valid density checkpoint')
    torch.save(dict(model=state),path/'best.pt');model.load_state_dict(state)
    (path/'fit.json').write_text(__import__('json').dumps(dict(history=history,best_step=beststep,steps=steps,selection='supported validation NLL; support failures explicitly reported, full NLL infinite when any excluded'),indent=2))
    return model.eval().requires_grad_(False)
