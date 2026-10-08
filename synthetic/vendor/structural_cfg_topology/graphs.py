"""The eight accepted graphs; indices are zero-based only inside Python."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Graph:
    name: str
    x_dim: int
    parents: tuple
    separator: tuple
    private: tuple
    edges: tuple = ()
    parent_graph: str | None = None

    @property
    def z_dim(self):
        return len(self.parents)

    @property
    def pair(self):
        return self.parent_graph or self.name

    def to_dict(self):
        return asdict(self)

    def validate(self):
        blocks = self.separator + tuple(i for block in self.private for i in block)
        if sorted(blocks) != list(range(self.x_dim)) or len(set(blocks)) != self.x_dim:
            raise ValueError(f"X partition is not disjoint/exhaustive: {self.name}")
        if len(self.private) != self.z_dim:
            raise ValueError("Every Z needs a private block, possibly empty")
        for parents, private in zip(self.parents, self.private):
            if not parents or not set(parents) <= set(self.separator + private):
                raise ValueError("A specification crosses private blocks")
        for a, b in self.edges:
            if a >= b or not any(a in block and b in block for block in (self.separator, *self.private)):
                raise ValueError("Feature edge violates this separator factorization")
        return self


def pairs(n):
    return tuple((2*j, 2*j+1) for j in range(n))


P3, P4, P5, P6 = pairs(3), pairs(4), pairs(5), pairs(6)
GRAPHS = (
    Graph("S8-DISJOINT", 8, ((0, 1, 2), (3, 4, 5), (6, 7)), (),
          ((0, 1, 2), (3, 4, 5), (6, 7))),
    Graph("S7-HUB", 7, tuple(p+(6,) for p in P3), (6,), P3),
    Graph("L10-WITHIN", 10, P5, (), P5, P5),
    Graph("L11-CHAIN", 11, (P4[0]+(8,), P4[1]+(8, 9), P4[2]+(9, 10), P4[3]+(10,)),
          (8, 9, 10), P4),
    Graph("L12-CORE", 12, tuple(p+(10, 11) for p in P5), (10, 11), P5),
    Graph("L14-ISLANDS", 14, tuple(p+((12,) if j < 3 else (13,)) for j, p in enumerate(P6)),
          (12, 13), P6),
    Graph("E14-BRIDGE", 14, tuple(p+((12,) if j < 3 else (13,)) for j, p in enumerate(P6)),
          (12, 13), P6, ((12, 13),), "L14-ISLANDS"),
    Graph("E12-NO-PRIVATE", 12, tuple(p+(10, 11) for p in P5)+((10, 11),),
          (10, 11), P5+((),), (), "L12-CORE"),
)
NAMES = tuple(g.name for g in GRAPHS)


def graph(name):
    return next(g for g in GRAPHS if g.name == name).validate()
