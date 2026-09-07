#!/usr/bin/env python
"""Train the Stage-2 editor under one of the three data strategies (§4.2, §8).

Examples:
  train_editor.py --strategy a --cache-dir data_cache
  train_editor.py --strategy b --pseudo-parallel-path runs/pseudo_parallel_filtered.json
  train_editor.py --strategy c --arm lewis --detector-dir runs/detector
"""
from __future__ import annotations

import argparse
import logging

from biasneut.common.config import EditorConfig, load_config, parse_cli_overrides
from biasneut.common.logging_utils import setup_logging
from biasneut.common.seeding import set_seed
from biasneut.data.cache_io import load_detection_examples, load_edit_examples
from biasneut.detector.infer import DetectorInference
from biasneut.editor.mask_infill import MaskInfiller
from biasneut.editor.train_strategy_a import run_strategy_a
from biasneut.editor.train_strategy_b import run_strategy_b
from biasneut.editor.train_strategy_c import run_strategy_c_lewis

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=["a", "b", "c"])
    parser.add_argument("--arm", default="lewis", choices=["lewis", "mask_infill"],
                         help="Strategy C only: which unsupervised arm to run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--pseudo-parallel-path", default=None, help="Strategy B: filtered pairs JSON")
    parser.add_argument("--detector-dir", default="runs/detector", help="Strategy C: trained Stage-1 detector")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    cfg = load_config(EditorConfig, args.config, {"strategy": args.strategy, **parse_cli_overrides(args.set)})

    if args.strategy == "a":
        wnc_train = load_edit_examples(f"{args.cache_dir}/wnc/train.json")
        wnc_dev = load_edit_examples(f"{args.cache_dir}/wnc/dev.json")
        editor = run_strategy_a(wnc_train, wnc_dev, cfg)
        editor.save(cfg.output_dir)

    elif args.strategy == "b":
        if not args.pseudo_parallel_path:
            raise SystemExit("--pseudo-parallel-path is required for Strategy B")
        pairs = load_edit_examples(args.pseudo_parallel_path)
        split = int(0.9 * len(pairs))
        editor = run_strategy_b(pairs[:split], pairs[split:], cfg)
        editor.save(cfg.output_dir)

    elif args.strategy == "c":
        detector = DetectorInference.from_pretrained(args.detector_dir)
        if args.arm == "mask_infill":
            infiller = MaskInfiller()
            infiller.save_dir = cfg.output_dir  # documented in README: no training needed for this arm
            logger.info("Strategy C / mask_infill needs no seq2seq training — use DetectorInference + "
                        "MaskInfiller directly at inference time (see biasneut.pipeline.MaskInfillEditorAdapter)")
        else:
            biased_train = [e for e in load_detection_examples(f"{args.cache_dir}/babe/train.json") if e.is_biased]
            biased_dev = [e for e in load_detection_examples(f"{args.cache_dir}/babe/dev.json") if e.is_biased]
            infiller = MaskInfiller()
            editor = run_strategy_c_lewis(biased_train, biased_dev, detector, infiller, cfg)
            editor.save(cfg.output_dir)

    logger.info("Done: strategy=%s output_dir=%s", args.strategy, cfg.output_dir)


if __name__ == "__main__":
    main()
