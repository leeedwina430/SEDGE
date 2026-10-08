"""Atomic, hash-checked artifacts for the new study only."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=json_default).encode()
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    value = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=json_default) + "\n")
    os.replace(temporary, path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def seal(directory, contract, files, **extra):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "complete.json", dict(
        status="complete", contract=contract, fingerprint=fingerprint(contract),
        files={name: sha256(directory / name) for name in files},
        completed_at=datetime.now(timezone.utc).isoformat(), **extra))


def completed(directory, contract=None):
    directory = Path(directory)
    path = directory / "complete.json"
    if not path.is_file():
        return False
    record = read_json(path)
    if (record.get("status") != "complete"
            or record["fingerprint"] != fingerprint(record["contract"])):
        raise RuntimeError(f"Invalid completion record: {directory}")
    if contract is not None and record["fingerprint"] != fingerprint(contract):
        raise RuntimeError(f"Artifact contract changed: {directory}")
    for name, expected in record["files"].items():
        if sha256(directory / name) != expected:
            raise RuntimeError(f"Artifact bytes changed: {directory / name}")
    return True


def code_hashes(*names):
    root = Path(__file__).resolve().parent
    paths = [root / name for name in names] if names else sorted(root.glob("*.py"))
    return {path.name: sha256(path) for path in paths}


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), message, flush=True)
