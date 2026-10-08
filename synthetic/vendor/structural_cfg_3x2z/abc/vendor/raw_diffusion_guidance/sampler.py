"""Ancestral updates, full clean-estimate DPS derivatives, and stable row-keyed noise."""
from __future__ import annotations
import time
import torch
from .common import keyed,tensor_hash


def noise_stream(cohort,seed,split,target_ids,d,steps=1000):
    # Each target owns a stream, independent of method, scale and evaluation batch.
    rows=[]
    for identity in target_ids:
        if torch.is_tensor(identity):identity=identity.item()
        g=torch.Generator().manual_seed(keyed('raw-guidance-v1',cohort,123+seed-42,split,str(identity),'noise'))
        rows.append(torch.randn((steps+1,d),generator=g))
    return torch.stack(rows,dim=1)


def dps_distance_gradient(prior,forward,x,t,z,calibration):
    x=x.detach().requires_grad_(True)
    eps=prior.predict_noise(x,t)
    a=prior.alpha_bars[t,None]
    clean=(x-(1-a).sqrt()*eps)/a.sqrt()
    r=forward(clean)+calibration['bias']-z
    r=torch.linalg.solve_triangular(calibration['cholesky'],r.T,upper=False).T
    distance=(r.square().sum(-1)+1e-12).sqrt()
    gradient=torch.autograd.grad(distance.sum(),x,create_graph=False)[0]
    return gradient.detach(),eps.detach(),distance.detach()


def reverse_step(model,x,t,z,noise,method,scale=0.,forward=None,density=None,calibration=None):
    tv=torch.full((len(x),),int(t),device=x.device,dtype=torch.long)
    gradient=None;distance=None
    if method=='dps' and scale!=0:
        with torch.enable_grad():gradient,eps,distance=dps_distance_gradient(model,forward,x,tv,z,calibration)
    else:
        with torch.no_grad():
            if method in ('cfg','cfg_s1'):
                if scale==0:eps=model.predict_noise(x,tv,z,False)
                elif scale==1:eps=model.predict_noise(x,tv,z,True)
                else:
                    e0=model.predict_noise(x,tv,z,False);e1=model.predict_noise(x,tv,z,True);eps=e0+scale*(e1-e0)
            else:eps=model.predict_noise(x,tv,z)
    with torch.no_grad():
        beta=model.betas[t];a=model.alphas[t];ab=model.alpha_bars[t];variance=model.posterior_variance[t]
        mean=(x-beta*eps/(1-ab).sqrt())/a.sqrt()
        proposal=mean+variance.sqrt()*noise if t>0 else mean
    if method=='dps' and scale!=0:proposal=proposal-scale*gradient
    if method=='cg' and scale!=0 and t>0:
        with torch.enable_grad():
            qx=x.detach().requires_grad_(True)
            gradient=torch.autograd.grad(density.log_prob(z,qx,tv).sum(),qx)[0].detach()
        proposal=proposal+scale*variance*gradient
    return proposal.detach(),gradient,distance


def sample(model,z,noise,method,scale=0.,forward=None,density=None,calibration=None,batch_size=1000):
    device=next(model.parameters()).device;cuda=device.type=='cuda';steps=model.steps
    assert noise.shape==(steps+1,len(z),model.d)
    if calibration:calibration={k:v.to(device) if torch.is_tensor(v) else v for k,v in calibration.items()}
    if cuda:torch.cuda.synchronize(device)
    began=time.perf_counter();out=[];diagnostics=[]
    for start in range(0,len(z),batch_size):
        stop=min(start+batch_size,len(z));c=z[start:stop].to(device);stream=noise[:,start:stop].to(device);x=stream[0].clone()
        for t in reversed(range(steps)):
            x,g,r=reverse_step(model,x,t,c,stream[t+1],method,scale,forward,density,calibration)
            if not bool(torch.isfinite(x).all()):raise FloatingPointError(f'Nonfinite {method} scale={scale} t={t} rows={start}:{stop}')
            if t%50==0 or t==steps-1:
                diagnostics.append(dict(step=t,row_start=start,row_stop=stop,x_abs_max=float(x.abs().max()),
                    gradient_norm_mean=float(g.norm(dim=-1).mean()) if g is not None else 0.,
                    gradient_abs_max=float(g.abs().max()) if g is not None else 0.,
                    residual_distance_mean=float(r.mean()) if r is not None else None))
        out.append(x.cpu())
    if cuda:torch.cuda.synchronize(device)
    return torch.cat(out),dict(sampling_seconds=time.perf_counter()-began,noise_sha256=tensor_hash(noise),trace=diagnostics)
