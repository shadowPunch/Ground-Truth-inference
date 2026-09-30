#!/usr/bin/env python
"""Full evaluation run (§6, §8 Phase 5): triad metrics + mandatory baselines
+ Pareto plot + significance tests, on a held-out test set.

Loads a *separate* "independent" detector checkpoint (never used for
training-time span localization or Strategy-B/C filtering) as required by
§4.3/§5.1/§6.1.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from biasneut.baselines.trivial import copy_input_baseline, delete_flagged_word_with_detector
from biasneut.common.config import EvalConfig, load_config, parse_cli_overrides
from biasneut.common.logging_utils import setup_logging
from biasneut.data.cache_io import load_detection_examples
from biasneut.detector.infer import DetectorInference
from biasneut.editor.seq2seq_model import SeqEditor
from biasneut.eval.fluency import FluencyScorer
from biasneut.eval.harness import EvaluationHarness, run_full_evaluation
from biasneut.eval.lexicon import LexiconBiasScorer
from biasneut.eval.preservation import PreservationScorer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--independent-detector-dir", default="runs/detector_eval_independent")
    parser.add_argument("--pipeline-detector-dir", default="runs/detector")
    parser.add_argument("--editor-dir", default="runs/editor")
    parser.add_argument("--lexicon-dir", default="data/lexicons")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(EvalConfig, args.config, parse_cli_overrides(args.set))

    test = load_detection_examples(f"{args.cache_dir}/babe/test.json")
    sources = [e.text for e in test]

    pipeline_detector = DetectorInference.from_pretrained(args.pipeline_detector_dir)
    editor = SeqEditor.load(args.editor_dir)
    predictions = pipeline_detector.predict(sources)
    word_tags = [p.word_tags for p in predictions]

    systems = {
        "copy_input": copy_input_baseline(sources),
        "delete_flagged_word": delete_flagged_word_with_detector(sources, pipeline_detector),
        "our_editor": editor.neutralize(sources, word_tags),
    }

    independent_detector = DetectorInference.from_pretrained(args.independent_detector_dir)
    lexicon_scorer = LexiconBiasScorer(args.lexicon_dir) if Path(args.lexicon_dir).exists() else None
    preservation_scorer = PreservationScorer(sbert_model=cfg.sbert_model, bertscore_model=cfg.bertscore_model)
    fluency_scorer = FluencyScorer(lm_name=cfg.fluency_lm, grammaticality_model=cfg.grammaticality_model)

    harness = EvaluationHarness(independent_detector, preservation_scorer, fluency_scorer, lexicon_scorer)
    results = harness.evaluate_all(systems, sources)

    run_full_evaluation(results, cfg.output_dir)
    logger.info("Evaluation complete: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
