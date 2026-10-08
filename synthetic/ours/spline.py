"""Conditional rational-quadratic CDF flow on a fixed learned interval.
Equations: Durkan et al., Neural Spline Flows (2019), arXiv:1906.04032.
CDF direction maps normalized observations to a uniform base distribution.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

def spline(value, raw, bins=16, inverse=False):
 value=value.double();raw=raw.double()
 widths=.001+(1-bins*.001)*raw[:,:bins].softmax(-1)
 heights=.001+(1-bins*.001)*raw[:,bins:2*bins].softmax(-1)
 slopes=.001+F.softplus(raw[:,2*bins:])
 xs=F.pad(widths.cumsum(-1),(1,0));ys=F.pad(heights.cumsum(-1),(1,0))
 # Fix endpoints exactly without changing interior knots.
 xs=torch.cat((xs[:,:-1],torch.ones_like(xs[:,-1:])),1);ys=torch.cat((ys[:,:-1],torch.ones_like(ys[:,-1:])),1)
 widths=xs[:,1:]-xs[:,:-1];heights=ys[:,1:]-ys[:,:-1]
 knots=ys if inverse else xs
 idx=((value[:,None]>=knots[:,1:]).sum(1)).clamp(max=bins-1)[:,None]
 def at(t):return t.gather(1,idx).squeeze(1)
 left,bottom,w,h,d0,d1=at(xs),at(ys),at(widths),at(heights),at(slopes),at(slopes[:,1:])
 secant=h/w;curvature=d0+d1-2*secant
 if inverse:
  residual=value-bottom
  qa=residual*curvature+h*(secant-d0);qb=h*d0-residual*curvature;qc=-secant*residual
  disc=qb.square()-4*qa*qc
  if bool((disc < -1e-12).any()):raise FloatingPointError('Negative spline discriminant')
  t=2*qc/(-qb-torch.sqrt(disc.clamp_min(0)))
 else:t=(value-left)/w
 middle=t*(1-t);denom=secant+curvature*middle
 numerator=secant.square()*(d1*t.square()+2*secant*middle+d0*(1-t).square())
 log_jac=numerator.log()-2*denom.log()
 out=left+w*t if inverse else bottom+h*(secant*t.square()+d0*middle)/denom
 return out, -log_jac if inverse else log_jac

class BoundedConditionalFlow(nn.Module):
 def __init__(self,fit,c_mean,c_std,bins=16,width=128):
  super().__init__();self.bins=bins
  self.register_buffer('bounds',torch.tensor([fit['alpha'],fit['beta'],fit['width']],dtype=torch.float64))
  self.register_buffer('c_location',torch.tensor([c_mean,c_std],dtype=torch.float64))
  self.net=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,3*bins+1))
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
  with torch.no_grad():self.net[-1].bias[2*bins:]=math.log(math.expm1(1-.001))
 def params(self,c):return self.net(((c.double()-self.c_location[0])/self.c_location[1]).float().reshape(-1,1))
 def interval(self,c):
  a=self.bounds[0]+self.bounds[1]*c.double();return a,a+self.bounds[2]
 def log_prob(self,z,c):
  a,b=self.interval(c);y=(z.double()-a)/(b-a);inside=(y>=0)&(y<=1)
  _,logdet=spline(y.clamp(0,1),self.params(c),self.bins)
  return torch.where(inside,logdet-(b-a).log(),torch.full_like(logdet,-torch.inf))
 def cdf(self,z,c):
  a,b=self.interval(c);y=(z.double()-a)/(b-a);u,_=spline(y.clamp(0,1),self.params(c),self.bins);return u
 def icdf(self,u,c):
  y,_=spline(u,self.params(c),self.bins,inverse=True);a,b=self.interval(c);return a+(b-a)*y
