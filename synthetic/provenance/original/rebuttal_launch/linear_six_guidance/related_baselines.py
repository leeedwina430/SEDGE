"""Explicit forward adaptations of three related works; never a theorem replication.

Training uses only archived hc fitting/validation rows. Sampling extensions and
proxy substitutions are recorded separately from the original papers' methods.
"""
from pathlib import Path
import argparse, copy, hashlib, json, math, time
import torch
from torch import nn
from run import ROOT, CASES, SETTINGS, paths, r

LABELS = {
    'rw_additive_mean': '[1] Additive · mean adaptation',
    'rw_additive_sample': '[1] Additive · residual sampling',
    'rw_rep4ex_mean': '[2] Rep4Ex-inspired · mean',
    'rw_rep4ex_sample': '[2] Rep4Ex-inspired · paired residual sampling',
    'rw_compositional_mean': '[3] Fixed soft composition · mean adaptation',
    'rw_compositional_sample': '[3] Fixed soft composition · residual sampling',
}
METHODS = list(LABELS)

def mlp(din, dout):
    return nn.Sequential(nn.Linear(din,64),nn.Softplus(),nn.Linear(64,64),nn.Softplus(),nn.Linear(64,dout))

class Forward(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.kind=kind; self.first=mlp(1,3); self.second=mlp(1,3)
    def forward(self,z):
        a,b=self.first(z[:,:1]),self.second(z[:,1:2])
        if self.kind=='additive': return a+b
        # Equation (8) of Wiedemer et al.; fixed, parameter-free composition.
        return torch.sigmoid(a)*a+torch.sigmoid(-a)*b

def state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def train_regression(model, inputs, target, vi, vt, steps, seed, label):
    rng=torch.Generator(device=target.device).manual_seed(seed)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3); best=math.inf; history=[]; start=time.monotonic()
    for step in range(1,steps+1):
        ix=torch.randint(len(target),(256,),device=target.device,generator=rng)
        loss=(model(inputs[ix])-target[ix]).square().mean()
        assert torch.isfinite(loss); opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad(): score=float((model(vi)-vt).square().mean())
            if score<best: best=score; best_step=step; saved=state(model)
            history.append(dict(step=step,train=float(loss),validation=score))
    model.load_state_dict(saved); model.eval()
    report=dict(best_step=best_step,validation_mse=best,updates=steps,seconds=time.monotonic()-start,history=history,parameters=sum(p.numel() for p in model.parameters()))
    print(label,report['seconds'],best,flush=True)
    return report

def affine(z,h):
    design=torch.cat((z,torch.ones_like(z[:,:1])),1)
    coef=torch.linalg.lstsq(design.double(),h.double()).solution.float()
    return coef, h-design@coef

def kernel(z, bandwidth): return torch.exp(-torch.cdist(z,z).square()/(2*bandwidth**2))

class Representation(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=mlp(3,2); self.decoder=mlp(2,3)
    def forward(self,x): return self.decoder(self.encoder(x))

class Control(nn.Module):
    def __init__(self):
        super().__init__(); self.nu=mlp(2,3); self.psi=mlp(2,3)
    def forward(self,hv): return self.nu(hv[:,:2])+self.psi(hv[:,2:])

def fit_rep(x,z,vx,vz,steps,seed):
    torch.manual_seed(seed); ae=Representation().to(x.device)
    rng=torch.Generator(device=x.device).manual_seed(seed+1)
    opt=torch.optim.Adam(ae.parameters(),lr=1e-3)
    bandwidth=torch.pdist(z[:512]).median().clamp_min(.1).detach()
    vk=kernel(vz[:512],bandwidth); history=[]; best=math.inf; start=time.monotonic()
    def objective(xx,zz,kk):
        h=ae.encoder(xx); _,res=affine(zz,h)
        recon=(ae.decoder(h)-xx).square().mean()
        mmr=(res*(kk@res)).sum()/(len(xx)**2*h.shape[1])
        hc=h-h.mean(0); cov=hc.T@hc/len(h)
        # Additional anti-collapse normalization, explicitly documented adaptation.
        scale=(cov-torch.eye(2,device=x.device)).square().mean()
        return recon+10*mmr+.1*scale, recon, mmr, scale
    for step in range(1,steps+1):
        ix=torch.randint(len(x),(256,),device=x.device,generator=rng)
        loss,*_=objective(x[ix],z[ix],kernel(z[ix],bandwidth))
        assert torch.isfinite(loss); opt.zero_grad();loss.backward();opt.step()
        if step%100==0 or step==steps:
            with torch.no_grad(): vals=objective(vx[:512],vz[:512],vk); score=float(vals[0])
            if score<best: best=score; best_step=step; saved=state(ae)
            history.append(dict(step=step,objective=score,reconstruction=float(vals[1]),mmr=float(vals[2]),whitening=float(vals[3])))
    ae.load_state_dict(saved); ae.eval()
    with torch.no_grad():
        h=ae.encoder(x); vh=ae.encoder(vx); coef,v=affine(z,h)
        vv=vh-torch.cat((vz,torch.ones_like(vz[:,:1])),1)@coef
    torch.manual_seed(seed+2); control=Control().to(x.device)
    cf=train_regression(control,torch.cat((h,v),1),x,torch.cat((vh,vv),1),vx,steps,seed+3,'control')
    with torch.no_grad():
        u=x-control.nu(h)
        corr=float((control.nu(h).mean(0)-x.mean(0)).square().mean())
        sv=torch.linalg.svdvals(coef[:2].double())
    report=dict(updates=steps,best_step=best_step,validation_objective=best,history=history,control=cf,
        seconds=time.monotonic()-start,affine_singular_values=sv.tolist(),offset_squared=corr,
        proxy='Outcome X reused as proxy; separate proxy absent',latent_dim=2,
        theorem_applicable=False,mmr_weight=10.,whitening_weight=.1,bandwidth=float(bandwidth),
        caveat='2D representation cannot invert full-dimensional 3D X; selected observational Z is not an exogenous action; bounded DGP does not establish full-support invariant residuals.')
    return ae,control,coef,h,v,u,report

@torch.no_grad()
def rep_mean(control,coef,z,v,u):
    outputs=[]
    for start in range(0,len(z),32):
        zz=z[start:start+32]; base=torch.cat((zz,torch.ones_like(zz[:,:1])),1)@coef
        hh=base[:,None,:]+v[None,:,:]
        outputs.append(control.nu(hh.reshape(-1,2)).reshape(len(zz),len(v),3).mean(1)+u.mean(0))
    return torch.cat(outputs)

def checksum(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(index,smoke=False,device='cuda',steps=3000):
    name,seed=CASES[index]; s=next(s for s in SETTINGS if s['name']==name)
    dest=ROOT/('related_smoke' if smoke else 'related_cells')/name/str(seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists(): return
    if smoke: steps=3
    torch.set_num_threads(1); torch.manual_seed(72000+seed)
    ip,rp,_=paths(s,seed);cohort=r.load_tensor(ip);norm=r.Normalizer(**cohort['normalizers']['hc'])
    train=cohort['arms']['hc']['train']; val=cohort['arms']['hc']['validation']
    def prepare(rows):
        return norm.normalize_x(rows['x']).float().to(device),norm.normalize_z(rows['z']).float().to(device)
    x,z=prepare(train); vx,vz=prepare(val)
    if smoke: x,z,vx,vz=x[:512],z[:512],vx[:128],vz[:128]
    fitted={};reports={};states={};start=time.monotonic()
    for kind in ['additive','compositional']:
        torch.manual_seed(72100+seed); model=Forward(kind).to(device)
        reports[kind]=train_regression(model,z,x,vz,vx,steps,72200+seed,kind)
        with torch.no_grad():
            residual=x-model(z); offset=residual.mean(0); residual=residual-offset
        fitted[kind]=(model,offset,residual); states[kind]=state(model)
    ae,cf,coef,h,v,u,reports['rep4ex']=fit_rep(x,z,vx,vz,steps,72300+seed)
    # Target values and reference X first loaded after every checkpoint is selected.
    ref=r.load_tensor(rp);ids=ref['ids'];tz=norm.normalize_z(ref['z']).float().to(device)
    if smoke: ids=ids[:12];tz=tz[:12]
    rng=torch.Generator().manual_seed(72400+seed)
    ix=torch.randint(len(x),(len(ids),),generator=rng).to(device)
    # Fixed fitting-only Monte Carlo bank, independent of targets and validation.
    bank=torch.randperm(len(x),generator=torch.Generator().manual_seed(72500+seed))[:min(2048,len(x))].to(device)
    out={}
    with torch.no_grad():
        for kind,(model,offset,residual) in fitted.items():
            prediction=model(tz)
            out[f'rw_{kind}_mean']=prediction
            out[f'rw_{kind}_sample']=prediction+residual[ix]
        out['rw_rep4ex_mean']=rep_mean(cf,coef,tz,v[bank],u[bank])
        th=torch.cat((tz,torch.ones_like(tz[:,:1])),1)@coef+v[ix]
        out['rw_rep4ex_sample']=cf.nu(th)+u[ix]
    # Architectural checks: exact additive cross-block mixed derivatives and
    # equality to the prescribed fixed compositor, independent of fit quality.
    probe=z[:8].detach().clone().requires_grad_(True)
    first=torch.autograd.grad(fitted['additive'][0](probe).sum(),probe,create_graph=True)[0][:,0]
    mixed=torch.autograd.grad(first.sum(),probe)[0][:,1]
    assert torch.equal(mixed,torch.zeros_like(mixed))
    comp=fitted['compositional'][0]
    with torch.no_grad():
        a,b=comp.first(probe[:,:1]),comp.second(probe[:,1:2])
        torch.testing.assert_close(comp(probe),torch.sigmoid(a)*a+torch.sigmoid(-a)*b,rtol=0,atol=0)
    errors={}
    for method,xx in out.items():
        xx=norm.inverse_x(xx.cpu().double()); assert torch.isfinite(xx).all()
        r.save_tensor(dest/f'{method}.pt',dict(x=xx,target_ids=ids,valid=torch.isfinite(xx).all(1)))
        errors[method]=float((norm.normalize_x(xx)-norm.normalize_x(ref['x'][:len(xx)].double())).square().mean())
    r.save_tensor(dest/'models.pt',dict(forward=states,ae=state(ae),control=state(cf),affine=coef.cpu(),normalizers=cohort['normalizers']['hc'],residual_indices=ix.cpu(),mc_indices=bank.cpu()))
    r.write_json(dest/'training.json',reports)
    r.write_json(dest/'checks.json',dict(additive_mixed_derivative_max=float(mixed.abs().max()),fixed_composition_exact=True,test_used_for_training=False,finite=True,heldout_single_draw_mse=errors))
    r.write_json(dest/'complete.json',dict(setting=name,seed=seed,methods=METHODS,targets=len(ids),smoke=smoke,steps=steps,seconds=time.monotonic()-start,source_sha256=checksum(__file__),inputs_sha256=checksum(ip),reference_sha256=checksum(rp),all_methods_are_adaptations=True))
    print('COMPLETE',name,seed,time.monotonic()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=3000)
    a=p.parse_args();main(a.index,a.smoke,a.device,a.steps)
