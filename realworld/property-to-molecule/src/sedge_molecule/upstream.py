"""Relocatable import bridge for the unchanged AnyMolGenCritic source closure.

The upstream modules use top-level ``data``, ``model`` and ``util`` imports.
Keeping those module names also supports historical vocabulary pickle objects.
The upstream model and graph grammar remain unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM_ROOT = (
    Path(__file__).resolve().parent
    / "_vendor" / "anymolgencritic"
)
UPSTREAM_SRC = UPSTREAM_ROOT / "src"


def _ensure_path() -> None:
    sys.dont_write_bytecode = True
    source_path = str(UPSTREAM_SRC)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


_ensure_path()

from data.target_data import (  # noqa: E402
    Data as TargetData,
    PAD_TOKEN,
    RING_START_TOKEN,
    RING_END_TOKENS,
    EOS_TOKEN_id,
)
from data.target_data import SpanningTreeVocabulary  # noqa: E402
from data.smiles import smiles_list2atoms_list, get_max_valence_from_dataset  # noqa: E402
from model.transformers_sota import Transformer, TransformerConfig  # noqa: E402
from model.generator import TokenEmbedding, EdgeLogitLayer  # noqa: E402

__all__ = [
    "UPSTREAM_ROOT", "UPSTREAM_SRC",
    "TargetData", "PAD_TOKEN", "RING_START_TOKEN", "RING_END_TOKENS", "EOS_TOKEN_id",
    "SpanningTreeVocabulary", "smiles_list2atoms_list", "get_max_valence_from_dataset",
    "Transformer", "TransformerConfig", "TokenEmbedding", "EdgeLogitLayer",
]
