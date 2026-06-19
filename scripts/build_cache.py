#!/usr/bin/env python3
"""Convenience script for monthly-noise cache generation."""

import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.cache_builder import DatasetCacheBuilder, parse_args


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")
    kwargs = vars(args)
    kwargs.pop("log_level", None)
    builder = DatasetCacheBuilder(**kwargs)
    builder.build()
