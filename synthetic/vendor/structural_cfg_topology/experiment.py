"""Frozen 60-cell comparison; generation is sealed before opening reference X."""
import argparse
import csv
import fcntl
from pathlib import Path
import shutil

import numpy as np
import torch
from raw_diffusion_guidance.common import environment
from .config import ARMS, METHODS, NAMES, PANELS, Config, configuration
from .data import build_cohort, calibrate
from .evaluation import repeated_rows, score, score_repeated
from .graphs import graph
from .io import (completed, fingerprint, load_tensor, log, read_json, save_tensor,
                 seal, sha256, write_json)
from .models import capacity, factors, make_model, training_tensors
from .runtime import configure_torch, seed_for
from .sampling import generate
from .training import optimize

ROOT = Path(__file__).resolve().parents[1]


def sources():
    paths = list((ROOT/"structural_cfg_topology").glob("*.py"))
    paths += [ROOT/"structural_cfg_topology"/s for s in ("run_slurm.sh", "source_provenance.json")]
    paths += list((ROOT/"structural_cfg_3x2z/abc/vendor").rglob("*.py"))
    paths += [ROOT/"structural_cfg_3x2z"/s for s in ("__init__.py", "core.py", "abc/__init__.py", "abc/vendor/provenance.json")]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def protocol(output):
    output = Path(output)
    p = read_json(output/"protocol.json")
    if p["sources"] != sources():
        raise ValueError("Frozen sources changed; use this run's source_snapshot")
    for key, hashes in p["data"].items():
        if sha256(output/"inputs"/(key+".pt")) != hashes["input_sha256"]:
            raise ValueError("Frozen training/target inputs changed")
    cfg = Config(**{**p["config"], "seeds": tuple(p["config"]["seeds"])})
    return p, cfg


def prepare(output, smoke=False):
    configure_torch()
    output = Path(output).resolve()
    if (output/"protocol.json").exists():
        p, cfg = protocol(output)
        if (cfg.mode == "smoke") != smoke:
            raise ValueError("Formal and smoke runs need separate directories")
        return p
    cfg = configuration(smoke)
    source = sources()
    calibration = calibrate(output/"references", cfg)
    data = {}
    for name in NAMES:
        for seed in cfg.seeds:
            cohort = build_cohort(output/"references", name, seed, cfg, calibration[name])
            inputs = {k: cohort[k] for k in ("contract", "fingerprint", "graph", "seed", "arms", "normalizers")}
            inputs["panels"] = {panel: {k: rows[k] for k in ("z", "ids")}
                                for panel, rows in cohort["panels"].items()}
            key = f"{name}/seed_{seed}"
            save_tensor(output/"inputs"/(key+".pt"), inputs)
            reference = output/"references/data"/key/"cohort.pt"
            data[key] = dict(input_sha256=sha256(output/"inputs"/(key+".pt")),
                             reference_sha256=sha256(reference))
    p = dict(config=cfg.to_dict(), sources=source, data=data,
        names=NAMES, methods=METHODS, arms=ARMS, panels=PANELS,
        capacity={name: {m: capacity(name, m, cfg) for m in METHODS} for name in NAMES},
        weights="all ones", condition_dropout=.1, early_stopping=False,
        checkpoint_selection="selected-validation best; EMA .999 for every DDPM",
        parameter_budget="~321435 across the entire deployed method; no unused networks",
        update_budget="20k per DDPM; 10k per MDN; CFG jointly updates all local networks 20k times",
        source_population="exact recovered topology RNG, counts, generation order and graph identifiers",
        evaluation="same three panels per replicate/arm; 32 x 32 independent conditional draws",
        no_reference_x_in_training_or_sampling=True, no_clipping_or_missing_target_replacement=True,
        precision="float32 models and shared linear schedule; float64 DGP, MDN weights and metrics",
        historical_results="old beta=.001→.2 and graph-dependent capacities are historical context only")
    for name in source:
        destination = output/"source_snapshot"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    write_json(output/"protocol.json", p)
    log(f"prepared {len(cells(p))} cells at {output}")
    return p


def cells(p):
    return [(g, a, m, s) for s in p["config"]["seeds"] for g in p["names"]
            for a in p["arms"] for m in p["methods"]]


def cell_directory(output, cell):
    name, arm, method, seed = cell
    return Path(output)/"cells"/name/arm/method/f"seed_{seed}"


def cell_contract(p, cell):
    return dict(protocol_fingerprint=fingerprint(p), cell=list(cell))


def load_models(directory, name, arm, method, seed, cfg, device, contract):
    models = {}
    for factor in factors(graph(name), method):
        state = load_tensor(directory/"training"/factor.name/"best.pt")
        if state["contract"] != dict(contract, factor=factor.to_dict()):
            # torch serialization preserves tuple fields in the factor contract.
            raise ValueError("Selected checkpoint has the wrong contract")
        model = make_model(name, method, factor, cfg, seed, device)
        model.load_state_dict(state["model"])
        models[factor.name] = model.eval().requires_grad_(False)
    return models


def generate_panels(data, arm, method, models, cfg):
    outputs = {}
    for panel, rows in data["panels"].items():
        chosen, repeated, draws = repeated_rows(data, rows, cfg)
        outputs[panel] = dict(main=generate(data, arm, method, models, rows, cfg), chosen=chosen,
            repeated=generate(data, arm, method, models, repeated, cfg, draw_ids=draws))
    return outputs


def check_samples(data, outputs, cfg):
    for panel in PANELS:
        rows, saved = data["panels"][panel], outputs[panel]
        chosen, repeated, draws = repeated_rows(data, rows, cfg)
        if chosen != saved["chosen"]:
            raise ValueError("Repeated target selection changed")
        for key, expected, expected_draws in (("main", rows, [0]*len(rows["ids"])),
                                               ("repeated", repeated, draws)):
            sample = saved[key]
            if sample["x"].shape != (len(expected["ids"]), graph(data["graph"]).x_dim):
                raise ValueError("Wrong generated shape")
            if (not torch.equal(sample["target_ids"], expected["ids"])
                    or sample["draw_ids"].tolist() != expected_draws):
                raise ValueError("Generated targets are not aligned")
            if not torch.equal(torch.isfinite(sample["x"]).all(1), sample["valid"]):
                raise ValueError("Failure mask does not match raw generated samples")


def verify_cell(output, p, cfg, cell):
    directory = cell_directory(output, cell)
    if not completed(directory, cell_contract(p, cell)):
        raise ValueError(f"Cell not complete: {cell}")
    receipt = read_json(directory/"complete.json")
    if receipt["replay_max_abs_error"] != 0:
        raise ValueError("Checkpoint replay failed")
    data = load_tensor(Path(output)/"inputs"/f"{cell[0]}/seed_{cell[3]}.pt")
    samples = load_tensor(directory/"samples.pt")
    check_samples(data, samples, cfg)
    return samples


def run_cell(output, name, arm, method, seed, device="cuda"):
    configure_torch()
    output = Path(output)
    p, cfg = protocol(output)
    cell = (name, arm, method, seed)
    if cell not in cells(p):
        raise ValueError("Cell outside the frozen experiment")
    directory, contract = cell_directory(output, cell), cell_contract(p, cell)
    if completed(directory, contract):
        verify_cell(output, p, cfg, cell)
        return
    data = load_tensor(output/"inputs"/f"{name}/seed_{seed}.pt")
    files, reports = ["samples.pt", "training.json"], []
    for factor in factors(graph(name), method):
        model = make_model(name, method, factor, cfg, seed, device)
        destination = directory/"training"/factor.name
        component_contract = dict(contract, factor=factor.to_dict())
        report = optimize(model, training_tensors(data, arm, factor), factor, cfg,
            seed_for(name, seed, method, factor.to_dict(), "fit"), destination, component_contract)
        if report["actual_updates"] != (cfg.density_updates if factor.family == "mdn" else cfg.diffusion_updates):
            raise ValueError("Fixed training budget was not reached")
        reports.append(dict(factor=factor.to_dict(), **report))
        files.extend(f"training/{factor.name}/{f}" for f in ("best.pt", "last.pt"))
        del model
    if sum(r["parameters"] for r in reports) != p["capacity"][name][method]["total"]:
        raise ValueError("Actual trained capacity differs")
    models = load_models(directory, name, arm, method, seed, cfg, device, contract)
    samples = generate_panels(data, arm, method, models, cfg)
    del models
    models = load_models(directory, name, arm, method, seed, cfg, device, contract)
    replay = generate_panels(data, arm, method, models, cfg)
    for panel in PANELS:
        for key in ("main", "repeated"):
            a, b = samples[panel][key], replay[panel][key]
            if not torch.equal(a["valid"], b["valid"]) or not torch.equal(torch.isnan(a["x"]), torch.isnan(b["x"])):
                raise ValueError("Replay changed failure masks")
            finite = torch.isfinite(a["x"])
            if not torch.equal(a["x"][finite], b["x"][finite]):
                raise ValueError("Checkpoint replay changed samples")
            if a["diagnostics"] != b["diagnostics"]:
                raise ValueError("Replay changed importance sampling diagnostics")
    check_samples(data, samples, cfg)
    save_tensor(directory/"samples.pt", samples)
    write_json(directory/"training.json", reports)
    seal(directory, contract, files, replay_max_abs_error=0., reference_x_read=False)
    log(f"COMPLETE {cell}")


def worker(output, worker_id, device="cuda"):
    output = Path(output)
    protocol(output)
    while True:
        with (output/"work_queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue = read_json(output/"work_queue.json")
            if not queue:
                return
            cell = queue.pop(0)
            write_json(output/"work_queue.json", queue)
        log(f"WORKER {worker_id} {cell}")
        run_cell(output, *cell, device=device)


def report(output):
    configure_torch()
    output = Path(output)
    p, cfg = protocol(output)
    # All 60 cells must finish before this process opens any reference X.
    verified = {cell: verify_cell(output, p, cfg, cell) for cell in cells(p)}
    write_json(output/"generation_complete.json", dict(protocol_fingerprint=fingerprint(p),
        cells=len(verified), artifacts={"/".join(map(str, cell)): sha256(cell_directory(output, cell)/"samples.pt")
                                      for cell in verified}))
    rows = []
    for name in NAMES:
        for seed in cfg.seeds:
            key = f"{name}/seed_{seed}"
            path = output/"references/data"/key/"cohort.pt"
            if sha256(path) != p["data"][key]["reference_sha256"]:
                raise ValueError("Evaluator reference changed")
            cohort = load_tensor(path)
            data = load_tensor(output/"inputs"/(key+".pt"))
            for arm in ARMS:
                for method in METHODS:
                    saved = verified[(name, arm, method, seed)]
                    for panel in PANELS:
                        reference = cohort["panels"][panel]
                        if any(not torch.equal(reference[k], data["panels"][panel][k]) for k in ("z", "ids")):
                            raise ValueError("Reference and target rows differ")
                        result = saved[panel]
                        rows.append(dict(graph=name, arm=arm, method=method, seed=seed, panel=panel,
                            parameters=p["capacity"][name][method]["total"],
                            main=score(cohort, reference, result["main"]),
                            conditional=score_repeated(cohort, reference, result["chosen"], result["repeated"], cfg)))
    write_json(output/"results.json", dict(rows=rows, protocol_fingerprint=fingerprint(p)))
    summaries = []
    for name in NAMES:
        for arm in ARMS:
            for panel in PANELS:
                for method in METHODS:
                    selected = [r for r in rows if (r["graph"], r["arm"], r["panel"], r["method"]) == (name, arm, panel, method)]
                    row = dict(graph=name, arm=arm, panel=panel, method=method, repeats=len(selected),
                        parameters=p["capacity"][name][method]["total"],
                        failed_main=sum(r["main"]["failed_count"] for r in selected),
                        failed_conditional_groups=sum(r["conditional"]["failed_groups"] for r in selected))
                    for label, section in (("mmd2_v", "main"), ("mmd2_paired_u", "main"),
                        ("swd_x_rms", "main"), ("support_rate", "main"), ("energy_score", "conditional")):
                        values = [r[section].get(label) for r in selected]
                        complete = all(v is not None for v in values)
                        row[label+"_mean"] = float(np.mean(values)) if complete else None
                        row[label+"_sd"] = (float(np.std(values, ddof=1)) if len(values)>1 else 0.) if complete else None
                    summaries.append(row)
    with (output/"summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    lines = ["# S7-HUB / L11-CHAIN：统一预算比较", "",
        "**Smoke 检查，不作科学结论。**" if cfg.mode != "formal" else "60 个模型单元完成固定训练预算、生成与完整 checkpoint 重放。",
        "", "linear100，β=1e-4→.02；每个 DDPM 20k，每个 MDN 10k；约32.1万总参数；无早停。",
        "原 DGP 的 IID/Hc 配对训练臂、5次数据重复、population/H/Hc 测试面板；三个方法共享测试条件。",
        "主指标为训练集固定核的 MMD² V；signed paired-U 另列。32×32独立条件生成使用配对真实 X 计算 energy score。",
        "任何目标失败均保留；失败面板和含失败重复的汇总为 NA，不计算成功样本子集均值。原样评分，无支持域裁剪。", "",
        "| DGP | 训练臂 | 测试 | 方法 | MMD² ↓ | SWD ↓ | Energy ↓ | 失败目标 |",
        "|---|---|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        def fmt(key):
            return "NA" if r[key+"_mean"] is None else f"{r[key+'_mean']:.5f} ± {r[key+'_sd']:.5f}"
        lines.append(f"| {r['graph']} | {r['arm']} | {r['panel']} | {r['method']} | {fmt('mmd2_v')} | {fmt('swd_x_rms')} | {fmt('energy_score')} | {r['failed_main']} |")
    lines += ["", "均值±样本标准差，非置信区间。IID/Hc 在数据重复内配对；各重复独立重新生成数据。",
        "参数量及每个因子的更新预算一致约束不代表总 FLOPs 或推理计算量相同。Neural21 采用有上限的 ESS 自适应 SIR。",
        "Structural CFG 在噪声时刻重组局部 score，不能据干净 Bayes 等价式断言生成等于精确后验。",
        "原 topology 的 β=.001→.2、不同参数量结果仅作历史参照。完整逐坐标、残差与失败记录见 results.json 和各单元 samples.pt。"]
    (output/"RESULTS_ZH.md").write_text("\n".join(lines)+"\n")
    seal(output/"evaluation", dict(protocol_fingerprint=fingerprint(p)),
         ["../results.json", "../summary.csv", "../RESULTS_ZH.md"], cells=len(verified),
         environment=environment(), full_sample_replay_max_abs_error=0.)
    log(f"report complete: {len(verified)} cells, {len(rows)} panel results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cell", "queue", "worker", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--graph", choices=NAMES)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.action == "prepare": prepare(args.out, args.smoke)
    elif args.action == "queue": write_json(args.out/"work_queue.json", cells(protocol(args.out)[0]))
    elif args.action == "worker": worker(args.out, args.worker, args.device)
    elif args.action == "cell": run_cell(args.out, args.graph, args.arm, args.method, args.seed, args.device)
    else: report(args.out)


if __name__ == "__main__":
    main()
