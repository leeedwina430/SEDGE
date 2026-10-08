"""CPU checks: exact saved data regeneration and density/DDPM numerical contracts."""
import json
from pathlib import Path
from common import runtime
import numpy as np
import torch
from datagen import progressive,multimodal
from ours.density import Density,BoundedFlow,time_context
from ours.spline import spline
from structural_cfg_topology.models import DDPM,diffusion_recipe
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    checks=[];reference_differences={}
    for suite,module in [('progressive',progressive),('multimodal',multimodal)]:
        for case in module.CASES:
            saved=torch.load(ROOT/'results/reported'/suite/'cells'/case['name']/'0/data.pt',map_location='cpu',weights_only=False)
            train,valid,truth,target=module.prepare(case,0,False)
            for split,regenerated in [('train',train),('validation',valid)]:
                for key in ['x','z']:assert torch.equal(saved[split][key],regenerated[key]),(case['name'],split,key)
            difference=float((saved['truth']-truth).abs().max())
            reference_differences[case['name']]=difference
            assert torch.allclose(saved['truth'],truth,rtol=0,atol=1e-14),case['name']
            assert torch.equal(saved['target'],target),case['name']
            assert np.allclose(module.mechanism(truth.numpy(),case),target.numpy())
            checks.append(case['name'])
            print('Exact training/validation/targets; reference max error:',case['name'],difference,flush=True)
    torch.manual_seed(19)
    y=torch.linspace(.001,.999,32,dtype=torch.float64)
    raw=torch.randn(32,49,dtype=torch.float64)
    z,ld=spline(y,raw);back,ild=spline(z,raw,inverse=True)
    assert torch.allclose(back,y,atol=1e-9)
    assert torch.allclose(ld+ild,torch.zeros_like(ld),atol=1e-8)
    for binary in [False,True]:
        model=Density(3,2,binary=binary,joint=True)
        c=torch.randn(16,3,requires_grad=True)
        labels=model.sample(c.detach())
        lp=model.log_prob(labels,c)
        g=torch.autograd.grad(lp.sum(),c)[0]
        assert torch.isfinite(lp).all() and torch.isfinite(g).all()
    c=torch.rand(128,1);labels=.6*c+.8*torch.rand_like(c)
    bounded=BoundedFlow(c,labels)
    assert torch.isfinite(bounded.log_prob(labels,c)).all()
    model=DDPM(2,1,'direct',192,diffusion_recipe(runtime.config(True)))
    x=torch.randn(16,2);t=torch.randint(100,(16,));ctx=torch.randn(16,1)
    assert torch.isfinite(model.predict_noise(model.corrupt(x,t,torch.randn_like(x)),t,ctx)).all()
    # Import complete training/sampling engines without any original workspace imports.
    from ours import engine
    from ours import multimodal as extension
    assert callable(engine.main) and callable(extension.main)
    result={'exact_train_validation_targets_seed_0':checks,'reference_max_abs_error':reference_differences,'reference_atol':1e-14,'numerical_checks':'passed','cuda_available':torch.cuda.is_available(),'full_training_rerun':False}
    (ROOT/'provenance/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
