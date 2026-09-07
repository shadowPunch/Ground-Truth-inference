#!/usr/bin/env python
"""Run the full detect-then-edit pipeline on a sentence or a file of
sentences (one per line)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from biasneut.common.logging_utils import setup_logging
from biasneut.detector.infer import DetectorInference
from biasneut.editor.seq2seq_model import SeqEditor
from biasneut.pipeline import BiasNeutralizationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-dir", default="runs/detector")
    parser.add_argument("--editor-dir", default="runs/editor")
    parser.add_argument("--sentence", default=None)
    parser.add_argument("--file", default=None, help="one sentence per line")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--no-constrained-decoding", action="store_true")
    args = parser.parse_args()

    setup_logging()

    if not args.sentence and not args.file:
        raise SystemExit("Provide --sentence or --file")

    sentences = [args.sentence] if args.sentence else read_lines(args.file)

    detector = DetectorInference.from_pretrained(args.detector_dir)
    editor = SeqEditor.load(args.editor_dir)
    pipeline = BiasNeutralizationPipeline(detector, editor)

    results = pipeline(
        sentences, num_beams=args.num_beams, use_constrained_decoding=not args.no_constrained_decoding,
    )
    for r in results:
        json.dump(
            {"source": r.source, "neutralized": r.neutralized, "was_flagged_biased": r.was_flagged_biased,
             "bias_prob": r.bias_prob, "flagged_span": r.flagged_span},
            sys.stdout,
        )
        sys.stdout.write("\n")


def read_lines(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    main()
