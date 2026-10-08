"""Fixed-budget AdamW, validation-selected EMA, and exact continuation."""
from copy import deepcopy
import math
from pathlib import Path
import time

import torch
from .io import fingerprint, load_tensor, log, save_tensor, write_json
from .models import StructuralCFG


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def optimize(model, tensors, factor, cfg, seed, destination, contract, stop_after=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    p = next(model.parameters())
    x, z, vx, vz = [v.to(p) for v in tensors]
    mdn = factor.family == "mdn"
    updates = cfg.density_updates if mdn else cfg.diffusion_updates
    interval = cfg.density_validate_every if mdn else cfg.validate_every
    rng = torch.Generator(device=p.device).manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.001 if mdn else .0003, weight_decay=0.)
    ema = None if mdn else deepcopy(model).eval().requires_grad_(False)
    parameters = list(model.parameters())
    ema_parameters = list(ema.parameters()) if ema is not None else None
    if not mdn:
        validation_rng = torch.Generator(device=p.device).manual_seed(seed+91003)
        vt = torch.randint(model.steps, (len(vx),), generator=validation_rng, device=p.device)
        vn = torch.randn(vx.shape, generator=validation_rng, device=p.device, dtype=p.dtype)

    @torch.no_grad()
    def validate(selected):
        selected.eval()
        value = 0.
        for start in range(0, len(vx), 512):
            a, b = vx[start:start+512], vz[start:start+512]
            if mdn:
                loss = -selected.log_prob(a, b).mean()
            else:
                t, n = vt[start:start+512], vn[start:start+512]
                xt = selected.corrupt(a, t, n)
                loss = (selected.component_loss(xt, t, n, b) if isinstance(selected, StructuralCFG)
                        else (selected.predict_noise(xt, t, b)-n).square().mean())
            value += float(loss)*len(a)/len(vx)
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss")
        return value

    start, best_step, elapsed, best, best_state = 0, 0, 0., math.inf, None
    history = []
    bound = fingerprint(dict(contract=contract, device=str(p.device), dtype=str(p.dtype)))
    last_path = destination/"last.pt"
    if last_path.exists():
        last = load_tensor(last_path)
        if last["fingerprint"] != bound:
            raise ValueError("Checkpoint contract changed")
        model.load_state_dict(last["model"])
        opt.load_state_dict(last["optimizer"])
        if ema is not None:
            ema.load_state_dict(last["ema"])
        rng.set_state(last["rng"])
        start, best_step, best, best_state = last["step"], last["best_step"], last["best"], last["best_state"]
        elapsed, history = last["elapsed_seconds"], last["history"]
    began = time.monotonic()

    def snapshot(step):
        save_tensor(last_path, dict(fingerprint=bound, step=step, model=cpu_state(model),
            optimizer=opt.state_dict(), ema=cpu_state(ema) if ema is not None else None,
            rng=rng.get_state(), best=best, best_state=best_state, best_step=best_step,
            elapsed_seconds=elapsed+time.monotonic()-began, history=history))

    log(f"fit {contract['cell']}/{factor.name} from {start} to {updates}")
    for step in range(start+1, updates+1):
        model.train()
        index = torch.randint(len(x), (min(512, len(x)),), generator=rng, device=p.device)
        loss = (-model.log_prob(x[index], z[index]).mean() if mdn
                else model.training_loss(x[index], z[index], rng))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 5.)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"Nonfinite gradient at {step}")
        opt.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_lerp_(ema_parameters, parameters, .001)
        if step % interval == 0 or step == updates:
            selected = ema if ema is not None else model
            score = validate(selected)
            if score < best-1e-5:
                best, best_step, best_state = score, step, cpu_state(selected)
            history.append(dict(step=step, train_loss=float(loss), validation_loss=score))
            write_json(destination/"progress.json", dict(step=step, best_step=best_step,
                best_validation_loss=best, elapsed_seconds=elapsed+time.monotonic()-began))
            snapshot(step)
        if step == stop_after:
            snapshot(step)
            return None
    if best_state is None or start > updates:
        raise ValueError("No valid selected checkpoint within the fixed budget")
    model.load_state_dict(best_state)
    model.eval()
    report = dict(actual_updates=updates, best_step=best_step, best_validation_loss=best,
        seconds=elapsed+time.monotonic()-began, parameters=sum(q.numel() for q in parameters),
        early_stopping=False, selection="best_validation" if mdn else "best_validation_ema",
        optimizer="AdamW", learning_rate=.001 if mdn else .0003, weight_decay=0.,
        batch_size=512, gradient_clip=5., ema=None if mdn else .999, history=history)
    save_tensor(destination/"best.pt", dict(model=best_state, contract=contract, report=report))
    return report
