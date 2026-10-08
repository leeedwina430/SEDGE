"""Historical training primitives and unchanged benchmark configuration."""
from pathlib import Path
import sys
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
from structural_cfg_topology.config import Config
from structural_cfg_topology.data import Normalizer
from structural_cfg_topology.training import optimize
from structural_cfg_topology.io import save_tensor,load_tensor,write_json

def config(smoke=False):
    return replace(Config(), seeds=tuple(range(5)), target_count=2000,
        **(dict(mode='smoke', diffusion_updates=2, density_updates=2,
                validate_every=1, density_validate_every=1) if smoke else {}))
