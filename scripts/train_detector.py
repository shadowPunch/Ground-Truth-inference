#!/usr/bin/env python
"""Train the Stage-1 detector (§5.1, §8 Phase 1).

Trains once on BABE+BASIL train splits. Run twice with different
``--output-dir``/``--seed`` values to get two *independent* checkpoints: one
used anywhere in the pipeline (span localization, Strategy-B/C filtering)
and a second, never touched by those, reserved purely for the Aggregate Bias
Score in ``scripts/run_evaluation.py`` (§4.3, §5.1, §6.1).
"""
from __future__ import annotations

import argparse
import logging

from transformers import AutoTokenizer

from biasneut.common.config import DetectorConfig, load_config, parse_cli_overrides
from biasneut.common.logging_utils import setup_logging
from biasneut.common.seeding import set_seed
from biasneut.data.cache_io import load_detection_examples
from biasneut.detector.train import save_pretrained, train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                         help="cap train/dev example counts, for fast dev-loop smoke runs")
    parser.add_argument("--set", nargs="*", default=[], help="key=value config overrides")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    cfg = load_config(DetectorConfig, args.config, parse_cli_overrides(args.set))

    train = load_detection_examples(f"{args.cache_dir}/babe/train.json") + \
        load_detection_examples(f"{args.cache_dir}/basil/train.json")
    dev = load_detection_examples(f"{args.cache_dir}/babe/dev.json") + \
        load_detection_examples(f"{args.cache_dir}/basil/dev.json")

    if args.limit:
        train, dev = train[: args.limit], dev[: max(1, args.limit // 4)]

    logger.info("Training detector on %d examples, evaluating on %d", len(train), len(dev))
    model = train_detector(train, dev, cfg, resume=not args.no_resume)

    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    save_pretrained(model, tokenizer, cfg.output_dir)
    logger.info("Saved detector to %s", cfg.output_dir)


if __name__ == "__main__":
    main()
