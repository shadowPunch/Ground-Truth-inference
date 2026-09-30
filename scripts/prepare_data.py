#!/usr/bin/env python
"""Acquire BABE/BASIL/WNC, build leakage-safe splits, and cache them as JSON
for the training scripts (§4, §6.5, Work-plan Phase 0)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from biasneut.data.babe import load_babe
from biasneut.data.basil import load_basil
from biasneut.data.splits import assert_no_leakage, group_disjoint_split
from biasneut.data.wnc import load_wnc
from biasneut.common.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _dump(examples, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([dataclasses.asdict(e) for e in examples], indent=None))
    logger.info("Wrote %d examples to %s", len(examples), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--wnc-dir", default="data/bias_data/WNC",
                         help="Path to the unzipped WNC release (see README: Data)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sources", nargs="+", default=["babe", "basil", "wnc"],
                         choices=["babe", "basil", "wnc"])
    args = parser.parse_args()

    setup_logging()
    cache_dir = Path(args.cache_dir)

    if "babe" in args.sources:
        logger.info("Loading BABE from Hugging Face Hub...")
        babe = load_babe()
        train, dev, test = group_disjoint_split(babe, seed=args.seed)
        assert_no_leakage(train, dev, test)
        _dump(train, cache_dir / "babe" / "train.json")
        _dump(dev, cache_dir / "babe" / "dev.json")
        _dump(test, cache_dir / "babe" / "test.json")

    if "basil" in args.sources:
        logger.info("Loading BASIL (shallow git clone if not cached)...")
        basil = load_basil(cache_dir)
        train, dev, test = group_disjoint_split(basil, seed=args.seed)
        assert_no_leakage(train, dev, test)
        _dump(train, cache_dir / "basil" / "train.json")
        _dump(dev, cache_dir / "basil" / "dev.json")
        _dump(test, cache_dir / "basil" / "test.json")

    if "wnc" in args.sources:
        wnc_dir = Path(args.wnc_dir)
        logger.info("Loading WNC from %s", wnc_dir)
        _dump(load_wnc(wnc_dir / "biased.word.train"), cache_dir / "wnc" / "train.json")
        _dump(load_wnc(wnc_dir / "biased.word.dev"), cache_dir / "wnc" / "dev.json")
        _dump(load_wnc(wnc_dir / "biased.word.test"), cache_dir / "wnc" / "test.json")


if __name__ == "__main__":
    main()
