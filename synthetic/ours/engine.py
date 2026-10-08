"""Progressive SEDGE benchmark: direct models, flow LWS/LGD, noisy-state CG."""
import argparse,json,math,statistics,sys,types,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from datagen.progressive import CASES,audit,prepare,mechanism,compatible,binary_probability
from common import runtime as r
from structural_cfg_topology import sampling as sp
from structural_cfg_topology.evaluation import mmd_squared
from structural_cfg_topology.models import DDPM,Factor,diffusion_recipe
from ours.density import Density,BoundedFlow,fit_density,time_context,load
ROOT=Path(__file__).resolve().parents[1]/'runs/progressive'

def fit_diff(name,x,z,vx,vz,cfg,dest,seed):
    torch.manual_seed(seed);m=DDPM(x.shape[1],z.shape[1],'prior' if z.shape[1]==0 else 'direct',192,diffusion_recipe(cfg)).cuda()
    r.optimize(m,(x,z,vx,vz),Factor(name,'ddpm'),cfg,seed,dest/name,dict(cell=[name,seed],recipe='sedge_benchmark_v1'))
    return m.eval().requires_grad_(False)

def draw(model,context,noise,score=None,composition=None,scale=1.):
    context=context.cuda();noise=noise.cuda();x=noise[0].clone()
    for offset,timestep in enumerate(reversed(range(model.steps)),1):
        t=torch.full((len(x),),timestep,device=x.device,dtype=torch.long)
        with torch.no_grad():
            eps=model.predict_noise(x,t,context)
            if composition is not None:eps=composition[0].predict_noise(x,t,context[:,:1])+composition[1].predict_noise(x,t,context[:,1:2])-eps
            mean=(x-model.betas[timestep]*eps/(1-model.alpha_bars[timestep]).sqrt())/model.alphas[timestep].sqrt()
        if score is not None:
            with torch.enable_grad():
                xx=x.detach().requires_grad_(True);lp=score(xx,t,context);g=torch.autograd.grad(lp.sum(),xx)[0]
            g=g*(50/g.norm(dim=1,keepdim=True).clamp_min(50)) # declared numerical stabilizer
            mean=mean+scale*model.betas[timestep]/model.alphas[timestep].sqrt()*g
        x=mean.detach()
        if timestep:x=x+model.posterior_variance[timestep].sqrt()*noise[offset]
        if not torch.isfinite(x).all():return torch.full_like(x.cpu(),float('nan'))
    return x.cpu()

def metrics(outputs,truth,target,norm,case,train,dest):
    truth=truth.double();target=target.double();ref=norm.normalize_x(truth);zz=norm.normalize_z(target)
    design=norm.normalize_x(train['x'][:512]);zt=norm.normalize_z(train['z'][:512]);kernels={}
    for name,features in [('x',design),('joint',torch.cat([design,zt],1))]:kernels[name]=dict(base_bandwidth=float(torch.pdist(features).median()),multipliers=[.5,1,2,4])
    refnn=cKDTree(ref.numpy());threshold=float(np.quantile(refnn.query(ref.numpy(),k=2)[0][:,1],.95));rows=[]
    for name,x in outputs.items():
        raw=x.double().numpy();validfinite=np.isfinite(raw).all(1);pred=mechanism(raw,case);rootlo=-1 if case['kind']=='binary' else 0;support=((raw>=rootlo)&(raw<=1)).all(1)
        sat=(pred==target.numpy()) if case['kind']=='binary' else (np.abs(pred-target.numpy())<=.005)
        c=raw[:,1:-1];one,two=compatible(c,target.numpy(),case);overlap=one&two&validfinite
        row=dict(method=name,n=len(x),finite_rate=float(validfinite.mean()),z1_satisfaction=float((sat[:,0]&support&validfinite).mean()),z2_satisfaction=float((sat[:,1]&support&validfinite).mean()),both_residuals=float((sat.all(1)&validfinite).mean()),validity=float((sat.all(1)&support&validfinite).mean()),bridge_overlap=float(overlap.mean()),overlap_and_validity=float((overlap&support&sat.all(1)).mean()))
        if validfinite.all():
            v=norm.normalize_x(x.double());row['mechanism_mae']=float(np.abs(pred-target.numpy()).mean())
            for metric,a,b in [('x',v,ref),('joint',torch.cat([v,zz],1),torch.cat([ref,zz],1))]:row[metric+'_mmd']=math.sqrt(max(0,mmd_squared(a,b,kernels[metric])['mmd2_v']))
            tree=cKDTree(v.numpy());coverage=tree.query(ref.numpy())[0];precision=refnn.query(v.numpy())[0]
            row.update(nn_coverage=float((coverage<=threshold).mean()),nn_reference_to_generated=float(coverage.mean()),nn_generated_to_reference=float(precision.mean()),nn_radius=threshold)
            # Repeated reference targets, fixed 32-draw panels (binary: one group).
            groups=[torch.arange(len(x))] if case['kind']=='binary' else [torch.arange(j,min(j+32,len(x))) for j in range(0,len(x),32)]
            cm=[math.sqrt(max(0,mmd_squared(v[ix],ref[ix],kernels['x'])['mmd2_v'])) for ix in groups if len(ix)>1];row['conditional_x_mmd']=statistics.mean(cm)
            row['diversity_mean_coordinate_std']=float(v.std(0).mean())
        rows.append(row)
    r.write_json(dest/'results.json',rows);return rows

def legacy_baselines(x,z,vx,vz,tz,steps,seed,dest):
    lg=load('sedge_legacy',Path(__file__).resolve().parents[1]/'baselines/adaptations.py')
    # Preserve the historical three architectures, extend X dimension only.
    original_mlp=lg.mlp;dimension=x.shape[1]
    lg.mlp=lambda din,dout:original_mlp(dimension if din==3 else din,dimension if dout==3 else dout)
    outputs={};reports={};torch.manual_seed(seed);ix=torch.randint(len(x),(len(tz),),device='cuda')
    for kind in ['additive','compositional']:
        model=lg.Forward(kind).cuda();reports[kind]=lg.train_regression(model,z,x,vz,vx,steps,seed,kind)
        with torch.no_grad():res=x-model(z);res=res-res.mean(0);prediction=model(tz)
        outputs[kind+'_mean']=prediction.cpu();outputs[kind+'_residual']= (prediction+res[ix]).cpu()
    ae,cf,coef,h,v,u,reports['rep4ex']=lg.fit_rep(x,z,vx,vz,steps,seed+1)
    with torch.no_grad():
        mean=[];bank=torch.randperm(len(x),device='cuda')[:2048]
        for start in range(0,len(tz),32):
            zz=tz[start:start+32];location=torch.cat([zz,torch.ones_like(zz[:,:1])],1)@coef
            hh=location[:,None,:]+v[bank][None,:,:];mean.append(cf.nu(hh.reshape(-1,2)).reshape(len(zz),len(bank),dimension).mean(1)+u[bank].mean(0))
        outputs['rep4ex_mean']=torch.cat(mean).cpu();hh=torch.cat([tz,torch.ones_like(tz[:,:1])],1)@coef+v[ix];outputs['rep4ex_residual']=(cf.nu(hh)+u[ix]).cpu()
    r.write_json(dest/'legacy_training.json',reports);return outputs

def main(index,smoke=False):
    case=CASES[index//3];seed=index%3;dest=ROOT/('smoke' if smoke else 'cells')/case['name']/str(seed);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists() and json.loads((dest/'complete.json').read_text()).get('implementation_version')==2:return
    torch.set_num_threads(1);torch.manual_seed(3200000+seed);began=time.monotonic()
    a=audit(case);assert a['accepted'],a;r.write_json(dest/'selection_audit.json',a)
    train,valid,truth,target=prepare(case,seed,smoke);norm=r.Normalizer.fit(train['x'],train['z']);cfg=r.config(smoke)
    x=norm.normalize_x(train['x']).float();z=norm.normalize_z(train['z']).float();vx=norm.normalize_x(valid['x']).float();vz=norm.normalize_z(valid['z']).float();tz=norm.normalize_z(target).float()
    binary=case['kind']=='binary';density_z=train['z'].float() if binary else z;density_vz=valid['z'].float() if binary else vz;density_tz=target.float() if binary else tz
    n,d=x.shape;bd=d-2;u=torch.cat([x[:,1:-1],x[:,:1]],1);vu=torch.cat([vx[:,1:-1],vx[:,:1]],1);empty=x[:,:0];vempty=vx[:,:0]
    r.save_tensor(dest/'data.pt',dict(train=train,validation=valid,truth=truth,target=target,normalizer=norm.state_dict()))
    recipes=dict(CD=(x,z,vx,vz),prior=(x,empty,vx,vempty),individual_Z1=(x,z[:,:1],vx,vz[:,:1]),individual_Z2=(x,z[:,1:],vx,vz[:,1:]),anchor=(u,z[:,:1],vu,vz[:,:1]),private=(x[:,-1:],torch.cat([x[:,1:-1],z[:,1:]],1),vx[:,-1:],torch.cat([vx[:,1:-1],vz[:,1:]],1)))
    diff={name:fit_diff(name,*tensors,cfg,dest/'training',3300000+seed+j*100) for j,(name,tensors) in enumerate(recipes.items())}
    c=x[:,1:-1];vc=vx[:,1:-1];steps=2 if smoke else 10000
    clean=Density(bd,binary=True) if binary else BoundedFlow(c,density_z[:,1:])
    clean=fit_density(clean,c,density_z[:,1:],vc,density_vz[:,1:],dest/'clean_likelihood',steps,seed).cuda()
    # Distill the clean likelihood into noisy-anchor likelihood: teacher samples
    # pair anchor states with fresh Z2, avoiding a supervised missing-(Z1,Z2) hole.
    anchor=diff['anchor']
    def anchor_corrupt(xx,zz):
        with torch.no_grad():label=clean.sample(xx[:,:bd]);tt=torch.randint(anchor.steps,(len(xx),),device='cuda');ut=anchor.corrupt(xx,tt,torch.randn_like(xx))
        return time_context(torch.cat([ut,zz[:,:1]],1),tt),label
    noisy_anchor=fit_density(Density(bd+1+1+16,binary=binary),u,z[:,:1],vu,vz[:,:1],dest/'anchor_noisy_likelihood',steps,seed+10,anchor_corrupt).cuda()
    # Full-X shared block: no private completion. Use a noisy-state density
    # rather than pretending noiseless p(Z2|X) has an ordinary clean density.
    dense_anchor=diff['individual_Z1']
    def dense_anchor_corrupt(xx,zz):
        tt=torch.randint(dense_anchor.steps,(len(xx),),device='cuda');xt=dense_anchor.corrupt(xx,tt,torch.randn_like(xx))
        return time_context(torch.cat([xt,zz[:,:1]],1),tt),zz[:,1:]
    dense_labels=torch.cat([z[:,:1],density_z[:,1:]],1);dense_vlabels=torch.cat([vz[:,:1],density_vz[:,1:]],1)
    dense_anchor_likelihood=fit_density(Density(d+1+16,binary=binary),x,dense_labels,vx,dense_vlabels,dest/'dense_anchor_noisy_likelihood',steps,seed+11,dense_anchor_corrupt).cuda()
    # Matched CG ablations: all use the SAME unconditional diffusion prior.
    prior=diff['prior'];cg={}
    for label,joint,sparse in [('sparse_factorized',False,True),('dense_factorized',False,False),('dense_joint',True,False)]:
        if sparse:
            parts=[]
            for j,scope in enumerate([list(range(d-1)),list(range(1,d))]):
                def corrupt(xx,zz,scope=scope,j=j):
                    tt=torch.randint(prior.steps,(len(xx),),device='cuda');xt=prior.corrupt(xx,tt,torch.randn_like(xx));return time_context(xt[:,scope],tt),zz[:,j:j+1]
                parts.append(fit_density(Density(len(scope)+16,binary=binary),x,density_z,vx,density_vz,dest/('CG_'+label+str(j)),steps,seed+20+j,corrupt).cuda())
            cg[label]=parts
        else:
            def corrupt(xx,zz):
                tt=torch.randint(prior.steps,(len(xx),),device='cuda');return time_context(prior.corrupt(xx,tt,torch.randn_like(xx)),tt),zz
            cg[label]=fit_density(Density(d+16,2,binary,joint),x,density_z,vx,density_vz,dest/('CG_'+label),steps,seed+30,corrupt).cuda()
    # Sampling: same noise across direct/CG variants, and same anchor banks for LWS controls.
    count=len(tz);noise=torch.randn(100,count,d);outputs={}
    outputs['CD']=draw(diff['CD'],tz,noise)
    outputs['compositional_CD']=draw(prior,tz,noise,composition=[diff['individual_Z1'],diff['individual_Z2']])
    W=8 if smoke else 1024;banks=[]
    for start in range(0,count,8):
        ctx=tz[start:start+8,:1].repeat_interleave(W,0);banks.append(sp.ddpm_samples(anchor,ctx,torch.randn(100,len(ctx),bd+1)).reshape(-1,W,bd+1))
    bank=torch.cat(banks);logs=[]
    with torch.no_grad():
        for start in range(0,count,16):
            b=bank[start:start+16];zz=density_tz[start:start+16,1:].repeat_interleave(W,0);logs.append(clean.log_prob(zz.cuda(),b[:,:,:bd].reshape(-1,bd).cuda()).cpu().reshape(len(b),W))
    lw=torch.cat(logs);uniform=torch.rand(count);private_noise=torch.randn(100,count,1);sampling=[]
    rawbank=(bank[:,:,:bd].double()*norm.x_std[1:-1]+norm.x_mean[1:-1]).reshape(-1,bd).numpy();repeated=target.repeat_interleave(W,0).numpy();onebank,twobank=compatible(rawbank,repeated,case)
    if binary:
        probability=binary_probability(rawbank[:,0],case);oracle=np.full_like(probability,-np.inf);positive=probability>0;oracle[positive]=np.log(probability[positive])
    else:
        from datagen.progressive import components
        _,_,scale2,_=components(rawbank,case);oracle=np.where(twobank,-np.log(scale2),-np.inf)
    oracle=torch.tensor(oracle.reshape(count,W));missing=float((~(onebank&twobank).reshape(count,W).any(1)).mean())
    for label,scores in [('SEDGE_LWS',lw),('SEDGE_oracle_weights',oracle),('SEDGE_unweighted',torch.zeros_like(lw))]:
        scores=scores.clone();failed=~torch.isfinite(scores).any(1);scores[failed]=0;weights=scores.softmax(1);ix=(weights.cumsum(1)<uniform[:,None]).sum(1).clamp_max(W-1);pair=bank[torch.arange(count),ix]
        private=sp.ddpm_samples(diff['private'],torch.cat([pair[:,:bd],tz[:,1:]],1),private_noise);out=torch.cat([pair[:,bd:],pair[:,:bd],private],1);out[failed]=float('nan');outputs[label]=out
        rawc=pair[:,:bd].double()*norm.x_std[1:-1]+norm.x_mean[1:-1];one,two=compatible(rawc.numpy(),target.numpy(),case)
        sampling.append(dict(method=label,ess=float((1/weights.square().sum(1)).mean()),empty_bank_fraction=float(failed.double().mean()),missing_compatible_bank_fraction=missing,selected_bridge_overlap=float((one&two).mean())))
    scales=[1.] if smoke else [.5,1.,2.]
    for scale in scales:
        def dense_anchor_score(xt,tt,context):return dense_anchor_likelihood.log_prob(density_tz[:,1:].cuda(),time_context(torch.cat([xt,context],1),tt))
        outputs[f'SEDGE_dense_LGD_s{scale:g}']=draw(dense_anchor,tz[:,:1],noise,score=dense_anchor_score,scale=scale)
        def anchor_score(ut,tt,context):return noisy_anchor.log_prob(density_tz[:,1:].cuda(),time_context(torch.cat([ut,context],1),tt))
        pair=draw(anchor,tz[:,:1],torch.randn(100,count,bd+1),score=anchor_score,scale=scale)
        if torch.isfinite(pair).all():
            private=sp.ddpm_samples(diff['private'],torch.cat([pair[:,:bd],tz[:,1:]],1),private_noise);outputs[f'SEDGE_LGD_s{scale:g}']=torch.cat([pair[:,bd:],pair[:,:bd],private],1)
        else:outputs[f'SEDGE_LGD_s{scale:g}']=torch.full((count,d),float('nan'))
        for label,predictor in cg.items():
            def score(xt,tt,context,predictor=predictor,label=label):
                zz=density_tz.cuda()
                if label=='sparse_factorized':return sum(pred.log_prob(zz[:,j:j+1],time_context(xt[:,scope],tt)) for j,(pred,scope) in enumerate(zip(predictor,[list(range(d-1)),list(range(1,d))])))
                return predictor.log_prob(zz,time_context(xt,tt))
            outputs[f'CG_{label}_s{scale:g}']=draw(prior,tz,noise,score=score,scale=scale)
    outputs.update(legacy_baselines(x.cuda(),z.cuda(),vx.cuda(),vz.cuda(),tz.cuda(),2 if smoke else 3000,seed+40,dest))
    raw={k:norm.inverse_x(v.double()) for k,v in outputs.items()}
    r.save_tensor(dest/'samples.pt',dict(outputs=raw));metrics(raw,truth,target,norm,case,train,dest);r.write_json(dest/'sampling.json',sampling)
    # Clean likelihood mean MAE is evaluated separately from density NLL.
    with torch.no_grad():
        cc=vc[:512].cuda();draws=torch.stack([clean.sample(cc) for _ in range(128)],0).mean(0).cpu()
        lp=clean.log_prob(density_vz[:,1:].cuda(),vc.cuda());likelihood_report=dict(validation_outside_support=int((~torch.isfinite(lp)).sum()),full_validation_nll=float(-lp.mean()) if torch.isfinite(lp).all() else None)
        if binary:
            rawc=valid['x'][:len(cc),1];oracle=torch.tensor(binary_probability(rawc.numpy(),case))
            # population reference, not selected conditional mean
            probability=clean.parts[0](cc).sigmoid().flatten().cpu().double();mae=float((probability-oracle).abs().mean());likelihood_report['probability_mae']=mae
        else:
            from datagen.progressive import components
            aa,bb,dd,ee=components(valid['x'][:len(cc),1:-1].numpy(),case);oracle=torch.tensor(ee+dd/2)
            prediction=draws[:,0].double()*norm.z_std[1]+norm.z_mean[1];mae=float((prediction-oracle).abs().mean())
            # Integrate over the union of true and learned supports, not a
            # truncated common-success region. Oracle used for diagnostics only.
            a,b=clean.interval(cc[:64]);a=(a.cpu()*norm.z_std[1]+norm.z_mean[1]);b=(b.cpu()*norm.z_std[1]+norm.z_mean[1]);lo=torch.minimum(a,torch.tensor(ee[:64]));hi=torch.maximum(b,torch.tensor(ee[:64]+dd[:64]));grid=lo[:,None]+(hi-lo)[:,None]*torch.linspace(0,1,1025)
            ct=cc[:64].repeat_interleave(1025,0);zt=((grid-norm.z_mean[1])/norm.z_std[1]).reshape(-1,1).float().cuda();estimated=(clean.log_prob(zt,ct).exp()/norm.z_std[1]).cpu().reshape(64,1025)
            exact=((grid>=torch.tensor(ee[:64])[:,None])&(grid<=torch.tensor(ee[:64]+dd[:64])[:,None])).double()/torch.tensor(dd[:64])[:,None]
            err=(estimated-exact).abs();likelihood_report.update(density_mae_on_support_union=float(err.mean()),population_conditional_tv=float((.5*torch.trapezoid(err,grid,dim=1)).mean()),density_quadrature_points=1025)
    r.write_json(dest/'likelihood_metrics.json',dict(**likelihood_report,population_conditional_mean_mae=mae,monte_carlo_mean_draws=128,n=len(cc),note='Mean MAE, density MAE and integrated TV are distinct; oracle density diagnostics use population factors.'))
    r.write_json(dest/'complete.json',dict(implementation_version=2,case=case,seed=seed,seconds=time.monotonic()-began,methods=list(raw),training_rows=len(x),evaluation_rows=count,parameters={k:sum(p.numel() for p in v.parameters()) for k,v in diff.items()},gradient_norm_clip=50,likelihood='Bernoulli for binary; conditional spline flows otherwise',scales=scales))

def report():
    rows=[]
    for path in (ROOT/'cells').glob('*/*/results.json'):
        for row in json.loads(path.read_text()):rows.append(dict(row,case=path.parent.parent.name,seed=int(path.parent.name)))
    r.write_json(ROOT/'results.json',rows);lines=['# Progressive SEDGE benchmark','','Historical binary reference violates the working 1% conditional criterion (5%); all other geometries audited before submission. No claim of guaranteed SEDGE success.','','| Case | Method | Seeds | Validity | X2 overlap | Z2 satisfaction | X MMD | Joint MMD | NN coverage |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for case in CASES:
        for method in dict.fromkeys(v['method'] for v in rows):
            rr=[v for v in rows if v['case']==case['name'] and v['method']==method]
            if not rr:continue
            def avg(key):return f'{statistics.mean(v[key] for v in rr):.5f}' if all(key in v for v in rr) else 'NA: failed outputs'
            lines.append('| '+' | '.join([case['name'],method,str(len(rr))]+[avg(k) for k in ['validity','bridge_overlap','z2_satisfaction','x_mmd','joint_mmd','nn_coverage']])+' |')
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--smoke',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
    report() if a.report else main(a.index,a.smoke)
