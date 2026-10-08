"""Matched ABC experiments, reusing the checksum-pinned historical baselines."""
import sys
from pathlib import Path

# The archived numerical modules retain their original absolute imports.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
