#!/usr/bin/env python3
"""Convenience script for Lightning training."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from training.train_quantile_lightning import main


if __name__ == "__main__":
    main()
