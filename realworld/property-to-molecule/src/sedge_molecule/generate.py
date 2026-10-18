"""Generate one molecule per frozen request, without CFG, filtering or refill."""
import argparse
from collections import Counter
from itertools import groupby
import hashlib
import json
import os
from pathlib import Path
import time


def sampling_batch_seed(panel, sampling_seed, batch_index):
    """Historical 63-bit batch RNG seed, preserving panel names and encoding."""
    payload = f"{panel}|{sampling_seed}|{batch_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def read_prefix(path, requests):
    if not path.exists():
        return []
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        if index >= len(requests):
            raise ValueError("Generation has more rows than requested")
        if any(row.get(key) != value for key, value in requests[index].items()):
            raise ValueError(f"Generation is not the frozen request prefix at row {index}")
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024, help="Frozen paper batch is 1024; other sizes change sampling")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--allow-partial-checkpoint", action="store_true", help="Allow a stopped training run for an explicit smoke check")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and thread count must be positive")
    import numpy as np
    import torch
    from .data import load_assets
    from .io import (atomic_json, read_json, seed_all, resolve_device,
                     environment, output_lock)
    from .model import ConditionedCGenerator
    from .policy import PROTOCOL
    from .requests import read_requests
    from .sampling import sample_batch

    device = resolve_device(args.device)
    torch.set_num_threads(args.threads)
    seed_all()
    manifest, vocab, scaler = load_assets(args.assets)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training = checkpoint["binding"]
    if training.get("schema") != "sedge_ccond_training_v2" or training.get("arm") != "S_STRUCT_CCOND":
        raise ValueError("Expected a checkpoint produced by this package's sedge-train")
    reached_budget = checkpoint["progress"]["update"] == training["total_updates"]
    complete = reached_budget and checkpoint.get("complete_verified") is True
    if reached_budget and not complete:
        raise ValueError("Final training validation is incomplete/failed; resume training to verify before generation")
    if not complete and not args.allow_partial_checkpoint:
        raise ValueError("Checkpoint is incomplete; --allow-partial-checkpoint is required for smoke checks")
    model = ConditionedCGenerator(vocab).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    requests = read_requests(args.requests)
    binding = dict(schema="sedge_ccond_generation_v2", arm="S_STRUCT_CCOND", request_count=len(requests),
                   training_update=checkpoint["progress"]["update"], training_complete=complete,
                   batch_size=args.batch_size, device=str(device), threads=torch.get_num_threads(), precision="float32",
                   guidance=1.0, cfg_enabled=False, temperature=1.0, top_k=0, best_out_of_k=1,
                   refill=False, max_len=256, n_correct=20,
                   sampling_rng_contract=PROTOCOL["sampling_rng_contract"],
                   environment=environment())
    path = args.output / "generations.jsonl"
    with output_lock(args.output):
        prior = read_prefix(path, requests)
        binding_path = args.output / "binding.json"
        if binding_path.exists():
            if read_json(binding_path) != binding:
                raise ValueError("Generation settings/batch/device/environment differ from this output directory")
        elif prior:
            raise ValueError("Existing generation has no binding")
        else:
            unexpected = [p.name for p in args.output.iterdir() if p.name != ".writer.lock"]
            if unexpected:
                raise ValueError(f"Refusing to reuse unbound nonempty output directory: {unexpected}")
        if any(r.get("arm") != "S_STRUCT_CCOND" for r in prior):
            raise ValueError("Output records belong to a different arm")
        atomic_json(binding_path, binding)
        done_path = args.output / "completion.json"
        if done_path.exists():
            done = read_json(done_path)
            if len(prior) != len(requests) or done["records_written"] != len(requests):
                raise ValueError("Completed generation was modified")
            print("Frozen inventory already complete; no resampling")
            return
        started, done, offset, new_records = time.monotonic(), len(prior), 0, 0
        stats = Counter()
        for (panel, sampling_seed), group in groupby(requests, key=lambda r: (r["panel"], r["sampling_seed"])):
            panel_rows = list(group)
            for batch_index, left in enumerate(range(0, len(panel_rows), args.batch_size)):
                chunk = panel_rows[left:left + args.batch_size]
                end = offset + len(chunk)
                if end <= done:
                    offset = end
                    continue
                seed = sampling_batch_seed(panel, sampling_seed, batch_index)
                generator = torch.Generator(device=device).manual_seed(seed)
                raw = np.array([[r["target_mol_logp"], r["target_tpsa"]] for r in chunk], dtype=np.float64)
                z = torch.from_numpy(scaler.transform(raw).astype(np.float32))
                observed = torch.ones((len(chunk), 2), dtype=torch.bool)
                generated, cost = sample_batch(model, "S_STRUCT_CCOND", vocab, z, observed, generator)
                if len(generated) != len(chunk):
                    raise RuntimeError("Sampler lost attempts")
                stats.update(cost)
                produced = [dict(**request, **result, batch_rng_seed=seed, arm="S_STRUCT_CCOND",
                                 guidance=1.0)
                            for request, result in zip(chunk, generated)]
                skip = max(0, done - offset)
                if prior[offset:offset + skip] != produced[:skip]:
                    raise ValueError("Partial batch replay differed; refusing to append")
                prior.extend(produced[skip:])
                new_records += len(produced) - skip
                temporary = path.with_suffix(".jsonl.tmp")
                with temporary.open("w") as handle:
                    for row in prior:
                        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                offset = end
                atomic_json(args.output / "progress.json", dict(records=offset, total=len(requests),
                            panel=panel, sampling_seed=sampling_seed,
                            elapsed_seconds=time.monotonic() - started, forward_passes_this_invocation=dict(stats)))
                print(f"Generated {offset}/{len(requests)} attempts", flush=True)
        if len(read_prefix(path, requests)) != len(requests):
            raise ValueError("Incomplete generation inventory")
        atomic_json(done_path, dict(status="completed", records_written=len(requests),
                    new_records_this_invocation=new_records, elapsed_seconds=time.monotonic() - started,
                    forward_passes_this_invocation=dict(stats),
                    numerical_paper_reproduction_verified=False))


if __name__ == "__main__":
    main()
