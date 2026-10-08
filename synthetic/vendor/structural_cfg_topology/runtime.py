"""Original topology RNG namespace, retained for exact population reproduction."""
import hashlib
import json
from raw_diffusion_guidance.common import setup as configure_torch


def seed_for(*parts):
    encoded = json.dumps(["x_z_topology_20260914_v1", *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63-1)
