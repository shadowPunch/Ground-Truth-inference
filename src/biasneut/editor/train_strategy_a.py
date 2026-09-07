"""Strategy A (§4.2): pretrain the editor on WNC, then optionally continue
training on in-domain news pairs (from Strategy B's pseudo-parallel data, or
Strategy C's synthesized pairs) for domain adaptation.

Run standalone, this measures how far pure Wikipedia transfer gets on news
(§6.3's "Pryzant off-the-shelf" baseline uses the pretrain-only checkpoint)."""
from __future__ import annotations

import logging

from biasneut.common.config import EditorConfig
from biasneut.data.schema import EditExample
from biasneut.editor.seq2seq_model import SeqEditor
from biasneut.editor.train_common import train_seq2seq_editor

logger = logging.getLogger(__name__)


def run_strategy_a(
    wnc_train: list[EditExample],
    wnc_dev: list[EditExample],
    cfg: EditorConfig,
    adapt_train: list[EditExample] | None = None,
    adapt_dev: list[EditExample] | None = None,
) -> SeqEditor:
    logger.info("Strategy A phase 1: WNC pretraining on %d pairs", len(wnc_train))
    editor = train_seq2seq_editor(wnc_train, wnc_dev, cfg)

    if adapt_train:
        logger.info("Strategy A phase 2: domain adaptation on %d in-domain pairs", len(adapt_train))
        adapt_cfg = EditorConfig(**{**cfg.__dict__, "output_dir": cfg.output_dir + "_adapted"})
        editor = train_seq2seq_editor(adapt_train, adapt_dev or wnc_dev, adapt_cfg, editor=editor)

    return editor
