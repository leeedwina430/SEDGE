"""The historical CCOND method, separated from machine-specific execution."""
from copy import deepcopy
import json
from pathlib import Path

PROTOCOL = json.loads((Path(__file__).parent / "configs/paper_protocol.json").read_text())
ARM = "S_STRUCT_CCOND"


def resolve_arm(arm):
    if arm != ARM:
        raise ValueError(f"Only {ARM} is included in this release")
    return deepcopy(PROTOCOL)


def training_config(path=None):
    config = {key: PROTOCOL[key] for key in (
        "seed", "epochs", "batch_size", "peak_lr", "betas", "weight_decay",
        "clip", "warmup", "min_lr_frac",
    )}
    config["precision"] = "bf16"
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown = supplied.keys() - config.keys()
        if unknown:
            raise ValueError(f"Unknown training settings: {sorted(unknown)}")
        config.update(supplied)
    for key in ("seed", "epochs", "batch_size", "warmup"):
        if type(config[key]) is not int or config[key] < (0 if key in ("seed", "warmup") else 1):
            raise ValueError(f"Invalid {key}")
    import math
    for key in ("peak_lr", "clip", "weight_decay", "min_lr_frac"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(f"Non-finite or nonnumeric {key}")
    if config["peak_lr"] <= 0 or config["clip"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("Learning rate/clip must be positive; weight decay must be nonnegative")
    if not 0 <= config["min_lr_frac"] <= 1:
        raise ValueError("min_lr_frac must be in [0, 1]")
    if len(config["betas"]) != 2 or any(not 0 <= b < 1 for b in config["betas"]):
        raise ValueError("Adam betas must be in [0, 1)")
    if config["precision"] not in ("bf16", "float32"):
        raise ValueError("precision must be bf16 or float32")
    return config
