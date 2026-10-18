"""Common native graph-state decoder, without retries or property filtering."""
import torch
from sedge_molecule.upstream import TargetData
from .policy import resolve_arm


@torch.no_grad()
def sampling_logits(model, arm, batch, z, observed, stats=None):
    if model.training or not bool(observed.all()):
        raise ValueError('Formal sampling requires eval mode and both requested conditions')
    if resolve_arm(arm)['cfg_enabled']:raise ValueError('CFG is forbidden')
    result=model.generation_logits(batch,z,observed)
    if stats is not None:
        stats['conditional_passes']+=1
        stats['c_conditioner_passes']=stats.get('c_conditioner_passes',0)+1
        stats['trunk_passes']=stats.get('trunk_passes',0)+1
    return result


@torch.no_grad()
def sample_batch(model,arm,vocab,z,observed,generator):
    if not bool(observed.all()):raise ValueError('Out-of-scope missing-condition generation')
    device = next(model.parameters()).device
    n=len(z)
    states=[TargetData(MAX_LEN=256,vocab=vocab,n_correct=20,allow_empty_bond=False,
                       tracker_id=i) for i in range(n)]
    active=list(range(n));traces=[[] for _ in range(n)]
    stats=dict(conditional_passes=0,null_passes=0)
    for step in range(256):
        if not active:break
        batch=tuple(t.to(device=device, non_blocking=True) for t in TargetData.collate([states[i].featurize() for i in active]))
        logits=sampling_logits(model,arm,batch,z[active].to(device),observed[active].to(device),stats)
        legal=torch.isfinite(logits).any(dim=-1)
        picks=torch.zeros(len(active),dtype=torch.long,device=device)
        if bool(legal.any()):
            probs=torch.softmax(logits[legal],dim=-1)
            picks[legal]=torch.multinomial(probs,1,generator=generator).squeeze(-1)
        legal=legal.cpu().tolist();picks=picks.cpu().tolist()
        remaining=[]
        for pos,index in enumerate(active):
            if not legal[pos]:
                states[index].ended=True;states[index].error='no_legal_action';continue
            token=picks[pos];traces[index].append(token);states[index].update(token)
            if not states[index].ended:remaining.append(index)
        active=remaining
    result=[]
    for i,state in enumerate(states):
        error=state.error
        if not state.ended:error='incomplete'
        smiles=None
        if error is None:
            try:smiles=state.to_smiles()
            except Exception as exc:error=f'serialization_error:{type(exc).__name__}:{exc}'
        result.append(dict(smiles=smiles,ended=bool(state.ended),error=error,
                           n_tokens=len(traces[i]),token_trace=traces[i]))
    if stats['null_passes'] or stats['conditional_passes']!=stats.get('c_conditioner_passes',0):
        raise AssertionError('Wrong conditional decoding path')
    return result,stats
