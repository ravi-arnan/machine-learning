#!/usr/bin/env python3
"""Train models and save artifacts for apps and API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stroke_ml.config import ARTIFACTS_DIR
from stroke_ml.models import train_artifacts


def main():
    parser = argparse.ArgumentParser(description="Train stroke risk models")
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--recall-target", type=float, default=0.80)
    parser.add_argument("--fn-cost", type=float, default=20.0)
    args = parser.parse_args()

    meta = train_artifacts(
        artifacts_dir=args.output,
        recall_target=args.recall_target,
        fn_cost=args.fn_cost,
    )
    print(json.dumps(meta, indent=2))
    print(f"\nArtifacts saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
