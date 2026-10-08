"""Audited progressive benchmarks. Mechanism-only selection; no model scores."""
import json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
CASES=[
 dict(name='discrete_squares_reference',kind='binary',overlap=.1,reference_exception=True),
 dict(name='discrete_squares_strict',kind='binary',overlap=.018),
 dict(name='linear',kind='linear',width=.05),
 dict(name='cubic',kind='cubic',width=.05),
 dict(name='smoothstep',kind='smoothstep',width=.05),
 dict(name='private_bridge_interaction',kind='interaction',width=.05),
 dict(name='two_dimensional_bridge',kind='two_dimensional',width=.05),
]

def components(c,case):
    kind=case['kind']; c=np.asarray(c)
    if kind=='two_dimensional':
        b1=.7*c[...,0]+.3*c[...,1];b2=1-(.3*c[...,0]+.7*c[...,1]);a1=.8+.15*c[...,0]*c[...,1];a2=.8+.15*(1-c[...,0])*(1-c[...,1])
    else:
        v=c[...,0];a1=np.full_like(v,.8);a2=np.full_like(v,.8);b1=.6*v
        b2=.6*(v if kind=='linear' else v**3 if kind=='cubic' else 3*v*v-2*v**3)
        if kind=='interaction':b1=v;b2=1-v;a1=.8*(1+.3*v);a2=.8*(1+.3*(1-v))
    return a1,b1,a2,b2

def mechanism(x,case):
    x=np.asarray(x)
    if case['kind']=='binary':
        z1=(x[:,0]>=0)&(x[:,0]<=1)&(x[:,1]>=0)&(x[:,1]<=1)
        z2=(x[:,1]>=case['overlap']-1)&(x[:,1]<=case['overlap'])&(x[:,-1]>=0)&(x[:,-1]<=1)
        return np.column_stack([z1,z2]).astype(float)
    a1,b1,a2,b2=components(x[:,1:-1],case)
    return np.column_stack([a1*x[:,0]+b1,a2*x[:,-1]+b2])

def bands(case):
    if case['kind']=='binary':return [[1,1],[1,1]]
    d=case['width'];a,b=.5-d/2,.5+d/2;k=case['kind']
    if k=='two_dimensional':center=np.array([.35,.35])
    elif k=='interaction':center=np.array([b,b])
    else: center=np.array([.6*b,.8+.6*(a if k=='linear' else a**3 if k=='cubic' else 3*a*a-2*a**3)])
    half=.03*d;return np.column_stack([center-half,center+half]).tolist()

def selected(z,case):
    box=np.array(bands(case));return ((z>=box[:,0])&(z<=box[:,1])).all(1)

def binary_probability(c,case):
    """Population P(Z2=1|C); used only for oracle diagnostics."""
    c=np.asarray(c).reshape(-1)
    return .5*((c>=case['overlap']-1)&(c<=case['overlap']))

def compatible(c,z,case):
    if case['kind']=='binary':
        v=c[:,0];return (v>=0)&(v<=1),(v>=case['overlap']-1)&(v<=case['overlap'])
    a,b,d,e=components(c,case)
    return (z[...,0]>=b)&(z[...,0]<=b+a),(z[...,1]>=e)&(z[...,1]<=e+d)

def audit(case,resolution=1001):
    if case['kind']=='binary':
        overlap=case['overlap'];p=overlap/2
        return dict(name=case['name'],individual_X2_ranges=[[0,1],[overlap-1,overlap]],joint_X2_range=[0,overlap],individual_bridge_width=[1,1],joint_bridge_width=overlap,conditional_exclusion=[p,p],overall_exclusion=overlap/8,passes_conditional_1pct=p<=.01,accepted=p<=.01 or case.get('reference_exception',False),reference_exception=case.get('reference_exception',False),tightness_ratio=overlap,method='analytic',root_support=[-1,1],caveat='Historical mandatory reference explicitly retains 5% conditional exclusion; not an assumption-compliant result.')
    dim=2 if case['kind']=='two_dimensional' else 1
    v=(np.arange(resolution)+.5)/resolution
    c=np.column_stack([q.ravel() for q in np.meshgrid(v,v,indexing='ij')]) if dim==2 else v[:,None]
    a,b,d,e=components(c,case);box=np.array(bands(case));qs=[]
    for j,(scale,offset) in enumerate([(a,b),(d,e)]):
        qs.append(np.clip(np.minimum(scale,box[j,1]-offset)-np.maximum(0,box[j,0]-offset),0,None)/scale)
    mass=float(np.mean(qs[0]*qs[1]));cond=[mass/float(q.mean()) for q in qs]
    measures=[[],[],[]];ranges=[[],[],[]]
    for z1 in np.linspace(*box[0],5):
        for z2 in np.linspace(*box[1],5):
            one,two=compatible(c,np.array([z1,z2]),case)
            for j,mask in enumerate([one,two,one&two]):
                measures[j].append(float(mask.mean()));ranges[j].append(np.column_stack([c[mask].min(0),c[mask].max(0)]).tolist() if mask.any() else None)
    ratio=max(measures[2])/min(min(measures[0]),min(measures[1]))
    # These witnesses ensure each entire marginal remains supported.
    # X3 near zero / X1 near one for same-direction; private roots near one
    # escape both low bands for opposing interaction and 2D mechanisms.
    return dict(name=case['name'],test_bands=box.tolist(),excluded_bands=box.tolist(),conditional_exclusion=cond,overall_exclusion=mass,individual_bridge_measure_ranges=[[min(measures[j]),max(measures[j])] for j in [0,1]],joint_bridge_measure_range=[min(measures[2]),max(measures[2])],center_ranges=dict(Z1=ranges[0][12],Z2=ranges[1][12],both=ranges[2][12]),all_query_ranges=dict(Z1=ranges[0],Z2=ranges[1],both=ranges[2]),tightness_ratio=ratio,passes_conditional_1pct=max(cond)<=.01,accepted=max(cond)<=.01 and ratio<=.25 and min(measures[2])>0,bridge_dimension=dim,root_support=[0,1],marginal_support_retained=True,method='analytic private-root integration and bridge quadrature',certified=False,grid_resolution=resolution,caveat='For 2D, ranges are coordinate projections, not a Cartesian feasible box; also report feasible area. No performance qualification.')

def prepare(case,seed,smoke=False):
    rng=np.random.default_rng(3100000+seed);dim=4 if case['kind']=='two_dimensional' else 3;binary=case['kind']=='binary'
    x=rng.uniform(-1 if binary else 0,1,(200000,dim));z=mechanism(x,case);ids=np.flatnonzero(~selected(z,case));rng.shuffle(ids);fit,valid=ids[:20000],ids[20000:22000]
    count=8 if smoke else 2000
    if binary:
        truth=rng.uniform(0,1,(count,dim));truth[:,1]*=case['overlap'];target=np.ones((count,2))
    else:
        box=np.array(bands(case));xx=[];zz=[];total=0
        # Joint rejection in (z,c), weighted by the private inverse Jacobian.
        # All scales >= .8, so .64/(a1*a2) <= 1.
        for _ in range(10000):
            c=rng.uniform(0,1,(100000,dim-2));t=rng.uniform(box[:,0],box[:,1],(len(c),2));a,b,d,e=components(c,case)
            u=(t[:,0]-b)/a;v=(t[:,1]-e)/d
            keep=(u>=0)&(u<=1)&(v>=0)&(v<=1)&(rng.random(len(c))<.64/(a*d))
            xx.append(np.column_stack([u,c,v])[keep]);zz.append(t[keep]);total+=int(keep.sum())
            if total>=count:break
        else:raise RuntimeError('Conditional rejection sampler exhausted')
        truth=np.concatenate(xx)[:count];target=np.concatenate(zz)[:count]
        # Repeated draws at the SAME target allow genuine conditional MMD.
        targets=target[:(4 if smoke else 64)];repeats=2 if smoke else 32;truths=[]
        for t in targets:
            collected=[];have=0
            for _ in range(10000):
                c=rng.uniform(0,1,(10000,dim-2));a,b,d,e=components(c,case);u=(t[0]-b)/a;v=(t[1]-e)/d
                keep=(u>=0)&(u<=1)&(v>=0)&(v<=1)&(rng.random(len(c))<.64/(a*d))
                collected.append(np.column_stack([u,c,v])[keep]);have+=int(keep.sum())
                if have>=repeats:break
            else:raise RuntimeError('Repeated reference sampler exhausted')
            truths.append(np.concatenate(collected)[:repeats])
        truth=np.concatenate(truths);target=np.repeat(targets,repeats,axis=0)
    assert selected(target,case).all() and np.allclose(mechanism(truth,case),target)
    train={k:torch.tensor(v,dtype=torch.float64) for k,v in dict(x=x[fit],z=z[fit]).items()};validation={k:torch.tensor(v,dtype=torch.float64) for k,v in dict(x=x[valid],z=z[valid]).items()}
    if smoke:train={k:v[:128] for k,v in train.items()};validation={k:v[:128] for k,v in validation.items()}
    return train,validation,torch.tensor(truth),torch.tensor(target)

if __name__=='__main__':
    audits=[]
    for case in CASES:
        a=audit(case);refined=audit(case,1401) if case['kind']!='binary' else a
        assert a['accepted'] and refined['accepted'],(a,refined)
        a['refined_conditional_exclusion']=refined['conditional_exclusion'];audits.append(a)
        print(case['name'],a.get('center_ranges',a.get('joint_X2_range')),a['conditional_exclusion'],a['tightness_ratio'],flush=True)
        prepare(case,0,True)
    (ROOT/'selection_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    print('All geometries and exact-reference samplers passed; historical reference exception labeled.')
