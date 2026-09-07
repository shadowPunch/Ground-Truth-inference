"""Strategy C orchestration (§4.2): fully unsupervised editing, two arms.

Arm 1 (mask-and-infill) needs no seq2seq training at all — Stage 1's
detector + a (optionally neutral-domain-adapted) MLM infiller are sufficient
at inference time; see ``biasneut.editor.mask_infill.MaskInfiller.infill``.

Arm 2 (LEWIS-lite) uses arm 1 to synthesize pseudo-parallel pairs at scale,
then trains the shared seq2seq editor on them, giving a second, independently
inspectable Strategy-C variant with (likely) better fluency at the cost of
losing arm 1's "no training at all" property.
"""
from __future__ import annotations

import logging

from biasneut.common.config import EditorConfig
from biasneut.data.schema import DetectionExample
from biasneut.detector.infer import DetectorInference
from biasneut.editor.lewis import synthesize_lewis_pairs
from biasneut.editor.mask_infill import MaskInfiller
from biasneut.editor.seq2seq_model import SeqEditor
from biasneut.editor.train_common import train_seq2seq_editor

logger = logging.getLogger(__name__)


def run_strategy_c_lewis(
    biased_train: list[DetectionExample],
    biased_dev: list[DetectionExample],
    detector: DetectorInference,
    infiller: MaskInfiller,
    cfg: EditorConfig,
) -> SeqEditor:
    train_pairs = synthesize_lewis_pairs(biased_train, detector, infiller)
    dev_pairs = synthesize_lewis_pairs(biased_dev, detector, infiller)
    return train_seq2seq_editor(train_pairs, dev_pairs, cfg)
