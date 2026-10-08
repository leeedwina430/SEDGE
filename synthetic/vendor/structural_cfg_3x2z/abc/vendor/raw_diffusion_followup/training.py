"""Two budget snapshots on one optimizer trajectory, with audited parent imports."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import time
import torch
from .common import *
from .models import make_model, training_data, training_loss, validation_values
from raw_diffusion_guidance.training import calibrate




def _state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def validation_inputs(job, vx):
    device=vx.device
    seed=(keyed(job['cohort'],job['seed'],'fixed-validation-noise') if job['family']=='raw'
          else keyed('followup-neural-validation',job['cohort'],job['seed'],job['role']))
    rng=torch.Generator(device=device).manual_seed(seed)
    times=(torch.arange(len(vx),device=device)%1000)[torch.randperm(len(vx),generator=rng,device=device)]
    noise=torch.randn(vx.shape,generator=rng,device=device,dtype=vx.dtype)
    return times * (job['steps'] or 1000) // 1000,noise


@torch.no_grad()
def validate(model, job, vx, vz, vt, vn):
    model.eval();pieces={}
    for start in range(0,len(vx),512):
        sl=slice(start,start+512)
        rows=validation_values(model,job,vx[sl],vz[sl],vt[sl],vn[sl])
        for key,value in rows.items():pieces.setdefault(key,[]).append(value)
    vectors={key:torch.cat(v) for key,v in pieces.items()}
    result={key:float(value.mean()) for key,value in vectors.items()}
    # Match the parent exactly: mix the means, rather than mean a float32 mixture.
    if job['kind']=='cfg':result['loss']=.9*result['conditional']+.1*result['unconditional']
    if job['kind']=='density':
        steps=job['steps']
        result['time_bucket_nll']=[float(vectors['loss'][(vt>=b*steps//10)&(vt<(b+1)*steps//10)].mean()) for b in range(10)]
    assert math.isfinite(result['loss']),f'Nonfinite validation: {job_key(job)}'
    return result


def optimize(job, data, directory, contract, *, device='cuda', parent=None,
             stop_after=None, output=None):
    """Keep exact raw-parent operations; record patience without interrupting fits."""
    setup();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=make_model(job,device);parameter=next(model.parameters());device=parameter.device
    x,z,vx,vz=[v.to(device=device,dtype=parameter.dtype) for v in training_data(job,data)]
    vt,vn=validation_inputs(job,vx)
    ema=deepcopy(model).eval().requires_grad_(False) if job['ema'] is not None else None
    opt=torch.optim.AdamW(model.parameters(),lr=job['lr'],weight_decay=0.)
    rng=torch.Generator(device=device).manual_seed(job['fit_seed'])
    min_delta=1e-5;best=math.inf;best_step=0;best_state=None;stale=0
    losses=[];validations=[];start=0;elapsed=0.;initial=None;first_patience=None
    inherited=None
    path=directory/'last.pt'
    if path.exists():
        last=load(path)
        assert last['contract_fingerprint']==fingerprint(contract)
    elif parent is not None:
        # The frozen new protocol attests the old artifact and contract, not a rewritten old hash.
        parent=Path(parent)
        last=load(verify_artifact(contract['parent']['last']))
        original=read_json(verify_artifact(contract['parent']['contract']))
        assert last['contract_fingerprint']==fingerprint(original)
        assert original['cohort']==job['cohort'] and original['seed']==job['seed'] and original['kind']==job['kind']
        assert original['initialization_seed']==job['init_seed'] and original['fit_seed']==job['fit_seed']
        assert job['family']=='raw' and job['steps'] in (0,1000)
        assert original['recipe']['lr']==job['lr'] and original['recipe']['ema']==job['ema']
        assert original['recipe']['batch_size']==512 and original['recipe']['validate_every']==job['validate_every']
        inherited=dict(parent_last=contract['parent']['last'],actual_start=last['step'],
                       selected_step=last['best_step'],old_stale=last['stale'],
                       former_early_stop=last['stale']>=30)
        write_json(directory/'parent_import.json',inherited)
    else:last=None
    if last is not None:
        model.load_state_dict(last['model'])
        if ema is not None:ema.load_state_dict(last['ema'])
        opt.load_state_dict(last['optimizer']);rng.set_state(last['rng'])
        start=last['step'];best=last['best'];best_step=last['best_step'];best_state=last['best_state']
        stale=last['stale'];losses=last['losses'];validations=last['validations'];initial=last['initial'];elapsed=last['elapsed']
        first_patience=last.get('first_patience_step');inherited=last.get('inherited',inherited)
        assert len(losses)==start
        if start>=job['max_budget']:assert start==job['max_budget']
    if initial is None:initial=validate(model,job,vx,vz,vt,vn)
    # Reconstruct the first historical patience event independently of the stopping policy.
    if first_patience is None:
        rb=math.inf;rs=0
        for row in validations:
            if row['loss']<rb-min_delta:rb=row['loss'];rs=0
            else:rs+=1
            if rs>=30:first_patience=row['step'];break
    if device.type=='cuda':torch.cuda.synchronize(device)
    began=time.perf_counter()
    parameters=list(model.parameters());ema_parameters=list(ema.parameters()) if ema is not None else None
    budgets={actual_budget(job,label):label for label in BUDGETS}

    def checkpoint(step):
        current_elapsed=elapsed+time.perf_counter()-began
        save(path,dict(contract_fingerprint=fingerprint(contract),step=step,model=_state(model),
                      ema=_state(ema) if ema is not None else None,optimizer=opt.state_dict(),rng=rng.get_state(),
                      best=best,best_step=best_step,best_state=best_state,stale=stale,first_patience_step=first_patience,
                      elapsed=current_elapsed,losses=losses,validations=validations,initial=initial,inherited=inherited))

    def snapshot(step):
        label=budgets[step];dest=directory/f'b{label}'
        if completed(dest):return
        assert best_state is not None
        dest.mkdir(parents=True,exist_ok=True)
        save(dest/'model.pt',dict(model=best_state,job=job,budget_label=label,actual_updates=step,
                                 contract_fingerprint=fingerprint(contract),best_step=best_step))
        r=dict(job=job,budget_label=label,actual_updates=step,best_step=best_step,best_validation_loss=best,
               initial_validation=initial,validation_history=list(validations),stale=stale,
               first_patience_step=first_patience,termination='fixed_budget_snapshot',
               elapsed_seconds=elapsed+time.perf_counter()-began,source_parent=inherited,
               fixed_validation_noise_sha256=tensor_hash(vn),fixed_validation_times_sha256=tensor_hash(vt),
               parameters=sum(p.numel() for p in parameters),environment=environment())
        files=['model.pt','training.json']
        if job['kind']=='forward':
            selected=deepcopy(model);selected.load_state_dict(best_state)
            cal_started=time.perf_counter();cal=calibrate(selected,vx,vz)
            save(dest/'calibration.pt',cal)
            write_json(dest/'calibration.json',{k:v.tolist() if torch.is_tensor(v) else v for k,v in cal.items()})
            r['calibration_seconds']=time.perf_counter()-cal_started
            files+=['calibration.pt','calibration.json']
        if job.get('parent_dir') and label==20000:
            original_best=load(verify_artifact(contract['parent']['checkpoint']))['model']
            r['selected_state_identical_to_parent']=(best_state.keys()==original_best.keys() and
                all(torch.equal(best_state[k],original_best[k]) for k in best_state))
        write_json(dest/'training.json',r)
        seal(dest,files,contract_fingerprint=fingerprint(contract),budget_label=label)
        print(f'SNAPSHOT {job_key(job)} b{label} best={best_step} loss={best:.7g}',flush=True)

    if start in budgets:snapshot(start)
    if start==0:checkpoint(0)
    for step in range(start+1,job['max_budget']+1):
        model.train();idx=torch.randint(len(x),(512,),generator=rng,device=device)
        loss=training_loss(model,job,x[idx],z[idx],rng)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite objective {job_key(job)} step {step}')
        opt.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(parameters,5.)
        if not torch.isfinite(grad):raise FloatingPointError(f'Nonfinite gradient {job_key(job)} step {step}')
        opt.step()
        if ema is not None:
            with torch.no_grad():torch._foreach_lerp_(ema_parameters,parameters,1-job['ema'])
        losses.append(float(loss.detach()))
        if step%job['validate_every']==0 or step in budgets:
            score=validate(ema if ema is not None else model,job,vx,vz,vt,vn)
            validations.append(dict(step=step,**score))
            if score['loss']<best-min_delta:best=score['loss'];best_step=step;best_state=_state(ema if ema is not None else model);stale=0
            else:stale+=1
            if stale>=30 and first_patience is None:first_patience=step
            write_json(directory/'progress.json',dict(step=step,best_step=best_step,best_validation_loss=best,
                latest_validation=score,stale=stale,first_patience_step=first_patience,
                elapsed_seconds=elapsed+time.perf_counter()-began))
            if step%5000==0:print(f'PROGRESS {job_key(job)} {step} best={best:.7g}',flush=True)
            if step%1000==0 or step in budgets:checkpoint(step)
        if step in budgets:snapshot(step)
        if stop_after==step:checkpoint(step);return None
    assert len(losses)==job['max_budget']
    if device.type=='cuda':torch.cuda.synchronize(device)
    checkpoint(len(losses))
    result=dict(job=job,actual_updates=len(losses),best_step=best_step,best_validation_loss=best,
                initial_validation=initial,validation_history=validations,first_patience_step=first_patience,
                elapsed_seconds=elapsed+time.perf_counter()-began,source_parent=inherited,
                fresh_update_count=len(losses)-(inherited['actual_start'] if inherited else 0),
                fixed_validation_noise_sha256=tensor_hash(vn),fixed_validation_times_sha256=tensor_hash(vt),
                environment=environment(),status='complete')
    write_json(directory/'training.json',result)
    return result

