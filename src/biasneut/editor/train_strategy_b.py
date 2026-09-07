"""Strategy B (§4.2): train the editor on an LLM-synthesized pseudo-parallel
corpus (see ``biasneut.data.pseudo_parallel`` for generation + the mandatory
§4.3 circularity mitigations, which must be applied *before* pairs reach this
function — this module only trains on whatever it's given)."""
from __future__ import annotations

import logging

from biasneut.common.config import EditorConfig
from biasneut.data.schema import EditExample
from biasneut.editor.seq2seq_model import SeqEditor
from biasneut.editor.train_common import train_seq2seq_editor

logger = logging.getLogger(__name__)


def run_strategy_b(
    pseudo_train: list[EditExample],
    pseudo_dev: list[EditExample],
    cfg: EditorConfig,
    warm_start: SeqEditor | None = None,
) -> SeqEditor:
    logger.info("Strategy B: training on %d filtered LLM-synthesized pairs", len(pseudo_train))
    if not pseudo_train:
        raise ValueError("No pseudo-parallel pairs survived filtering — check §4.3 thresholds")
    return train_seq2seq_editor(pseudo_train, pseudo_dev, cfg, editor=warm_start)
