"""Immutable recipes, atomic local artifacts, and tensor-only input contracts."""
from __future__ import annotations
import hashlib,json,os,platform,random
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/raw_diffusion_guidance_20260913_v1'
COHORTS=('narrow','wide','cov_all')
SEEDS=(42,43,44,45,46)
KINDS=('prior','forward','density','cfg')
METHODS=('dps','cg','cfg')
GRIDS={'dps':[0,.01,.03,.1,.3,1,3],'cg':[0,.1,.3,1,2,3,5],'cfg':[0,.5,1,1.25,1.5,2,3]}
RECIPE=dict(timesteps=1000,beta_start=1e-4,beta_end=.02,time_dim=64,depth=3,
    updates=20000,batch_size=512,lr=3e-4,weight_decay=0.,ema=.999,validate_every=250,
    patience=30,min_delta=1e-5,train_gradient_clip=5.,condition_dropout=.1,
    density_components=4,density_cholesky_floor=.001,density_covariance_floor=1e-4)
WIDTHS={3:dict(direct=224,cfg=224,prior=212,forward=126,density=106),
        9:dict(direct=345,cfg=345,prior=327,forward=191,density=164)}
COUNTS={3:dict(direct=320547,cfg=320771,prior=287687,forward=32762,density=32460),
        9:dict(direct=749004,cfg=749349,prior=673302,forward=76022,density=76156)}

def read_json(p):return json.loads(Path(p).read_text())


def write_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,p)


def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    torch.save(value,tmp);os.replace(tmp,p)


def load(p):return torch.load(p,map_location='cpu',weights_only=False)


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def keyed(*parts):return int(fingerprint(list(parts))[:15],16)%(2**31-1)


def artifact(p):return dict(path=str(Path(p).resolve()),sha256=digest(p),bytes=Path(p).stat().st_size)


def verify_artifact(a):
    p=Path(a['path']);assert digest(p)==a['sha256'],f'Artifact changed: {p}';return p


def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def setup(threads=1):
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def environment():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,tf32=False)

