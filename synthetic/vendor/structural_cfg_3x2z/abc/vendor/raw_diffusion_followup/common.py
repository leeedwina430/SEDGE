from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import torch
from raw_diffusion_guidance.common import *
OUT=ROOT/'work'
TIMES=(100,1000)
BUDGETS=(20000,40000)
@lru_cache(maxsize=15)
def input_data(cohort,seed):
    entry=read_json(ROOT/'data/inputs.json')[cohort][str(seed)]
    path=ROOT/entry['path']
    assert digest(path)==entry['sha256'],'Input data changed'
    data=load(path)
    for key in ('x_mean','x_std','z_mean','z_std'):
        data[key]=data[key].flatten().float()
    return data

def job_key(job):
    return f"{job['family']}/{job['cohort']}/seed_{job['seed']}/{job['role']}/T{job['steps']}"


def job_dir(output, job):
    return Path(output) / 'models' / job_key(job)


def snapshot_dir(output, job, budget_label):
    assert budget_label in BUDGETS
    return job_dir(output, job) / f'b{budget_label}'


def actual_budget(job, budget_label):
    assert budget_label in BUDGETS
    return job['base_budget'] * (budget_label // 20000)


def seal(directory, files, **metadata):
    directory=Path(directory)
    r=dict(**metadata,artifacts={f:artifact(directory/f) for f in files})
    write_json(directory/'complete.json',r)
    return r


def completed(directory):
    p=Path(directory)/'complete.json'
    if not p.exists(): return False
    r=read_json(p)
    for a in r['artifacts'].values():verify_artifact(a)
    return r

