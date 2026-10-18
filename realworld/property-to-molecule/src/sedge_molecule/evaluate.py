"""Evaluate a frozen request manifest, retaining every planned attempt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", default="v0-2", help="v0-2 or a profile JSON path")
    args = parser.parse_args(argv)
    from .evaluation.adapter import evaluate_files

    result = evaluate_files(args.requests, args.generations, args.train_csv,
                            args.output, profile=args.profile)
    print(json.dumps({"output": str(args.output), "panels": {
        panel: metrics["counts"] for panel, metrics in result["panels"].items()
    }}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
