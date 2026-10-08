from dataclasses import asdict, dataclass, replace

VERSION = "structural_cfg_topology_linear100_20k_v1"
NAMES = ("S7-HUB", "L11-CHAIN")
METHODS = ("structural_cfg", "direct_cd", "neural21")
ARMS = ("iid", "hc")
PANELS = ("population", "h", "hc")
TARGET_PARAMETERS = 321435


@dataclass(frozen=True)
class Config:
    mode: str = "formal"
    seeds: tuple = (42, 43, 44, 45, 46)
    train_count: int = 40000
    validation_count: int = 10000
    target_count: int = 1000
    design_count: int = 200000
    audit_count: int = 200000
    repeated_targets: int = 32
    repeated_draws: int = 32
    diffusion_updates: int = 20000
    density_updates: int = 10000
    validate_every: int = 250
    density_validate_every: int = 100
    timesteps: int = 100
    beta_start: float = .0001
    beta_end: float = .02
    candidate_initial: int = 64
    candidate_max: int = 16384
    target_ess: float = 32.
    target_batch: int = 16

    def to_dict(self):
        return asdict(self)


def configuration(smoke=False):
    return (replace(Config(), mode="smoke", seeds=(42,), train_count=64,
                    validation_count=32, target_count=4, repeated_targets=2,
                    repeated_draws=3, diffusion_updates=2, density_updates=2,
                    validate_every=1, density_validate_every=1)
            if smoke else Config())
