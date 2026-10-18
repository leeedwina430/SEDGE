"""Train S_STRUCT_CCOND from scratch with resumable checkpoints."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time


def lr_at(step, config, total_updates):
    if step < config["warmup"]:
        return config["peak_lr"] * (step + 1) / config["warmup"]
    progress = (step - config["warmup"]) / max(1, total_updates - config["warmup"])
    return config["peak_lr"] * (config["min_lr_frac"] + (1 - config["min_lr_frac"]) *
                               0.5 * (1 + math.cos(math.pi * progress)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON training settings; defaults match the paper")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:INDEX")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, default=0, help="Stop at this total update count (0: complete configured budget)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.num_workers < 0 or args.threads < 1 or args.max_updates < 0 or args.log_every < 1:
        parser.error("Invalid worker/thread/update/log setting")

    import torch
    from sedge_molecule.losses import total_loss
    from sedge_molecule.upstream import PAD_TOKEN
    from .data import load_assets, read_train_csv, make_loader
    from .model import ConditionedCGenerator
    from .policy import PROTOCOL, training_config
    from .io import (read_json, atomic_json, atomic_save, seed_all, resolve_device,
                     environment, get_rng, set_rng, output_lock)

    config = training_config(args.config)
    device = resolve_device(args.device)
    if config["precision"] == "bf16":
        supported = False
        if device.type == "cuda":
            with torch.cuda.device(device):
                supported = torch.cuda.is_bf16_supported()
        if not supported:
            raise ValueError("Paper BF16 training requires a BF16-capable CUDA device; use a float32 config for CPU checks")
    torch.set_num_threads(args.threads)
    seed_all(config["seed"], args.deterministic)
    manifest, vocab, scaler = load_assets(args.assets)
    if config["seed"] != manifest["seed"]:
        raise ValueError("Training seed must match prepared scratch initialization")
    split = read_train_csv(args.train_csv, custom=manifest["dataset_mode"] == "custom")
    if split.rows != manifest["train_rows"]:
        raise ValueError("Training row count differs from prepared assets")
    per_epoch = math.ceil(split.rows / config["batch_size"])
    updates = config["epochs"] * per_epoch
    model = ConditionedCGenerator(vocab).to(device)
    model.load_state_dict(torch.load(args.assets / "initial_model.pt", map_location="cpu", weights_only=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["peak_lr"], betas=tuple(config["betas"]),
                                 weight_decay=config["weight_decay"])
    binding = dict(schema="sedge_ccond_training_v2", arm="S_STRUCT_CCOND", config=config,
                   training_rng_contract=PROTOCOL["training_rng_contract"],
                   train_rows=split.rows,
                   dataset_mode=manifest["dataset_mode"], model_config=model.cfg.to_dict(),
                   c_conditioner=PROTOCOL["c_conditioner"], total_updates=updates,
                   device_type=device.type, threads=torch.get_num_threads(), deterministic=args.deterministic,
                   environment=environment())
    last = args.output / "checkpoints/last.pt"
    state = dict(update=0, epoch=0, batch_in_epoch=0)
    masks = Counter()
    with output_lock(args.output):
        if args.resume:
            if not last.is_file():
                raise ValueError("No checkpoint to resume")
            checkpoint = torch.load(last, map_location="cpu", weights_only=False)
            if checkpoint["binding"] != binding:
                changed = [k for k in binding if checkpoint["binding"].get(k) != binding[k]]
                raise ValueError(f"Resume inputs/settings/environment changed: {changed}")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            state = checkpoint["progress"]
            masks = Counter(checkpoint["mask_counts"])
            set_rng(checkpoint["rng"])
        elif any(p.name != ".writer.lock" for p in args.output.iterdir()):
            raise ValueError("Refusing to overwrite nonempty training output; use --resume with unchanged inputs")
        if args.max_updates and args.max_updates <= state["update"]:
            raise ValueError("max-updates must exceed the current checkpoint update")
        atomic_json(args.output / "binding.json", binding)
        started = time.monotonic()
        start_update = state["update"]
        verified_complete = False

        def save(path):
            return atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    progress=dict(state), binding=binding, rng=get_rng(),
                                    mask_counts=dict(masks),
                                    complete_verified=verified_complete), path)

        model.train()
        try:
            with (args.output / "train_log.jsonl").open("a") as log:
                while state["epoch"] < config["epochs"]:
                    loader = make_loader(split, vocab, scaler, epoch=state["epoch"],
                                         batch_size=config["batch_size"], start_batch=state["batch_in_epoch"],
                                         num_workers=args.num_workers, seed=config["seed"])
                    for batch, z, observed, rows in loader:
                        if not bool(observed.all()):
                            raise ValueError("CCOND requires both properties observed")
                        batch = tuple(t.to(device, non_blocking=True) for t in batch)
                        z, observed = z.to(device), observed.to(device)
                        lr = lr_at(state["update"], config, updates)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        optimizer.zero_grad(set_to_none=True)
                        precision = torch.autocast("cuda", dtype=torch.bfloat16) if config["precision"] == "bf16" else nullcontext()
                        with precision:
                            outputs = model(batch, z, observed)
                        loss, parts = total_loss(outputs, batch[0], z, vocab.get_id(PAD_TOKEN))
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Non-finite training loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
                        if not bool(torch.isfinite(norm)):
                            raise RuntimeError("Non-finite gradient norm")
                        optimizer.step()
                        masks["11"] += len(rows)
                        state["update"] += 1
                        state["batch_in_epoch"] += 1
                        if state["update"] % args.log_every == 0 or state["update"] == 1:
                            record = dict(update=state["update"], epoch=state["epoch"], lr=lr,
                                          grad_norm=float(norm),
                                          mask_counts=dict(masks), elapsed_seconds=time.monotonic() - started, **parts)
                            log.write(json.dumps(record, allow_nan=False) + "\n")
                            log.flush()
                            print(json.dumps(record), flush=True)
                        if state["batch_in_epoch"] == per_epoch:
                            state["epoch"] += 1
                            state["batch_in_epoch"] = 0
                            save(last)
                        if args.max_updates and state["update"] >= args.max_updates:
                            break
                    if args.max_updates and state["update"] >= args.max_updates:
                        break
            complete = state["update"] == updates
            if complete and (state["epoch"] != config["epochs"] or sum(masks.values()) != split.rows * config["epochs"]):
                raise ValueError("Training budget or exposure count is inconsistent")
            paper_settings = manifest["dataset_mode"] == "paper_train" and config == training_config()
            verified_complete = complete
            save(last)
            if complete:
                save(args.output / "checkpoints/final.pt")
            atomic_json(args.output / "completion.json", dict(
                status="completed" if complete else "stopped", progress=state,
                updates_this_invocation=state["update"] - start_update,
                elapsed_seconds=time.monotonic() - started, mask_counts=dict(masks),
                paper_training_settings_match=paper_settings,
                numerical_paper_reproduction_verified=False,
            ))
        except Exception as exc:
            atomic_json(args.output / "completion.json", dict(status="failed", error=repr(exc), progress=state,
                        recovery="Resume from the last committed checkpoint; in-flight batches were not saved"))
            raise


if __name__ == "__main__":
    main()
