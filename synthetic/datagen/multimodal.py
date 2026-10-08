"""Three binary multimodal candidates, two declared assumption regimes."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parent
CASES=[dict(name=f'multimodal_{variant}_{regime}',kind='binary',variant=variant,private_width=width,regime=regime) for variant in ['balanced','weighted','curved'] for regime,width in [('strict',.08),('reference',.5)]]

def masks(c):
    c=np.asarray(c);return ((c>=-1)&(c<=-.4))|((c>=.4)&(c<=1)),(c>=-.5)&(c<=.5)

def windows(c,case):
    c=np.asarray(c);w=case['private_width'];a=np.zeros_like(c);b=np.zeros_like(c);w1=np.full_like(c,w,dtype=float);w2=w1.copy()
    if case['variant']=='weighted':w2=w*np.where(c<0,.2,1.)
    elif case['variant']=='curved':
        a=.45*np.sin(np.pi*c);b=.45*np.sin(2*np.pi*c);w1=w*(.6+.4*c*c);w2=w*(.2+.8*((c+1)/2)**2)
    return a-w1/2,a+w1/2,b-w2/2,b+w2/2

def mechanism(x,case):
    x=np.asarray(x);c=x[:,1];a,b=masks(c);lo1,hi1,lo2,hi2=windows(c,case)
    return np.column_stack([a&(x[:,0]>=lo1)&(x[:,0]<=hi1),b&(x[:,2]>=lo2)&(x[:,2]<=hi2)]).astype(float)

def compatible(c,z,case):return masks(np.asarray(c)[:,0])

def binary_probability(c,case):
    c=np.asarray(c);_,b=masks(c);_,_,lo,hi=windows(c,case)
    return b*(np.minimum(1,hi)-np.maximum(-1,lo)).clip(0)/2

def probabilities(c,case):
    a,b=masks(c);lo1,hi1,lo2,hi2=windows(c,case)
    return a*(hi1-lo1)/2,b*(hi2-lo2)/2

def audit(case,resolution=None):
    cuts=[-1,-.5,-.4,0,.4,.5,1]
    def integral(fun):return sum(quad(fun,a,b,epsabs=1e-12)[0] for a,b in zip(cuts[:-1],cuts[1:]))/2
    p1=integral(lambda c:float(probabilities(c,case)[0]));p2=integral(lambda c:float(probabilities(c,case)[1]));mass=integral(lambda c:float(np.prod(probabilities(c,case))))
    right=quad(lambda c:float(np.prod(probabilities(c,case))),.4,.5,epsabs=1e-12)[0]/2/mass
    cond=[mass/p1,mass/p2];threshold=.01 if case['regime']=='strict' else .0500000001
    return dict(name=case['name'],individual_X2_sets=dict(Z1=[[-1,-.4],[.4,1]],Z2=[[-.5,.5]]),joint_X2_set=[[-.5,-.4],[.4,.5]],individual_widths=[1.2,1.],joint_width=.2,each_mode_width=.1,tightness_ratio=.2,conditional_exclusion=cond,overall_exclusion=mass,marginal_positive_probability=[p1,p2],posterior_right_mode_probability=right,passes_conditional_1pct=max(cond)<=.01,accepted=max(cond)<=threshold,acceptance_threshold=threshold,reference_exception=case['regime']=='reference',method='piecewise adaptive integration of exact private-window probabilities',root_support=[-1,1],note='Two 0.1-wide modes. Reference regime is explicitly not 1%-compliant; its maximum conditional exclusion is bounded by 5%. No performance filter.')

def prepare(case,seed,smoke=False):
    rng=np.random.default_rng(4100000+seed);x=rng.uniform(-1,1,(200000,3));z=mechanism(x,case);ids=np.flatnonzero(~z.astype(bool).all(1));rng.shuffle(ids);fit,val=ids[:20000],ids[20000:22000]
    n=8 if smoke else 2000;collected=[];count=0
    for _ in range(10000):
        # Equal-width overlap intervals. Acceptance yields density proportional
        # to the product of compatible private-window lengths.
        c=-.5+.1*rng.random(10000);c=np.where(rng.random(len(c))<.5,c,c+.9)
        lo1,hi1,lo2,hi2=windows(c,case);accept=rng.random(len(c))<(hi1-lo1)*(hi2-lo2)/case['private_width']**2
        xx=np.column_stack([rng.uniform(lo1,hi1),c,rng.uniform(lo2,hi2)]);collected.append(xx[accept]);count+=int(accept.sum())
        if count>=n:break
    else:raise RuntimeError('Conditional reference rejection failed')
    truth=np.concatenate(collected)[:n];target=np.ones((n,2));assert mechanism(truth,case).all() and (abs(truth)<=1).all()
    train=dict(x=torch.tensor(x[fit]),z=torch.tensor(z[fit]));validation=dict(x=torch.tensor(x[val]),z=torch.tensor(z[val]))
    if smoke:train={k:v[:128] for k,v in train.items()};validation={k:v[:128] for k,v in validation.items()}
    return train,validation,torch.tensor(truth),torch.tensor(target)

if __name__=='__main__':
    audits=[]
    for case in CASES:
        a=audit(case);assert a['accepted'],a
        train,_,truth,_=prepare(case,0);right=float((truth[:,1]>0).double().mean());assert abs(right-a['posterior_right_mode_probability'])<.05
        a['reference_sample_right_mode_fraction']=right;a['example_training_label_counts']={str(label):int(((train['z']==torch.tensor(label)).all(1)).sum()) for label in [[0,0],[1,0],[0,1],[1,1]]}
        assert a['example_training_label_counts']['[1, 1]']==0
        audits.append(a);print(case['name'],a['conditional_exclusion'],'right_mode',a['posterior_right_mode_probability'],'counts',a['example_training_label_counts'],flush=True)
    (ROOT/'selection_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
