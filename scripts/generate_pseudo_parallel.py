#!/usr/bin/env python
"""Strategy B, offline step (§4.2, §4.3): generate LLM-synthesized
biased->neutral pairs from BABE/BASIL positives, then apply the mandatory
circularity mitigations before anything is allowed to train an editor.

Runs with ``--dry-run`` (EchoClient, no API key, no network) so the
generation/filtering wiring can be exercised without spending API credits;
swap in ``AnthropicClient`` (and any other ``LLMClient`` you add) for a real
run, ideally with >=2 distinct model families per §4.3.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from biasneut.common.logging_utils import setup_logging
from biasneut.data.cache_io import load_detection_examples
from biasneut.data.pseudo_parallel import (
    AnthropicClient,
    EchoClient,
    filter_pseudo_parallel,
    generate_pseudo_parallel,
    sample_for_human_review,
)
from biasneut.detector.infer import DetectorInference
from biasneut.eval.preservation import PreservationScorer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--detector-dir", default="runs/detector",
                         help="Independent detector instance used for bias-drop filtering (§4.3)")
    parser.add_argument("--output", default="runs/pseudo_parallel_filtered.json")
    parser.add_argument("--similarity-floor", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="cap number of source sentences (dev/testing)")
    args = parser.parse_args()

    setup_logging()

    examples = load_detection_examples(f"{args.cache_dir}/babe/train.json")
    examples = [e for e in examples if e.is_biased]
    if args.limit:
        examples = examples[: args.limit]

    clients = [EchoClient()] if args.dry_run else [AnthropicClient()]
    logger.info("Generating with %d client(s) over %d biased sentences", len(clients), len(examples))
    pairs = generate_pseudo_parallel(examples, clients)

    detector = DetectorInference.from_pretrained(args.detector_dir)
    preservation_scorer = PreservationScorer()

    filtered = filter_pseudo_parallel(
        pairs,
        bias_score_fn=detector.bias_score,
        similarity_fn=lambda s, t: preservation_scorer._sbert.similarity(
            preservation_scorer._sbert.encode(s), preservation_scorer._sbert.encode(t)
        ).item(),
        similarity_floor=args.similarity_floor,
        require_bias_drop=not args.dry_run,  # EchoClient returns the input unchanged, so bias never "drops"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([dataclasses.asdict(p) for p in filtered]))
    logger.info("Wrote %d filtered pairs to %s", len(filtered), output_path)

    review_path = sample_for_human_review(filtered, out_path=output_path.with_suffix(".review.csv"))
    logger.info("Human spot-check sample written to %s (§4.3 mitigation 4)", review_path)


if __name__ == "__main__":
    main()
