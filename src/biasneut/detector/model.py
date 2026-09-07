"""Stage-1 detector (§5.1): shared encoder + sentence head + BIO token head.

Backbone defaults to POLITICS (`launch/POLITICS`, a roberta-base-scale
encoder pretrained by contrasting same-story articles across the ideological
spectrum — note this is roberta-BASE scale in the actual released checkpoint,
125M params, not roberta-large as sometimes described) with plain
DistilRoBERTa as a lighter fallback. Both are small enough to fully
fine-tune on a single free GPU (§5.1), so no PEFT is used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from biasneut.data.collate import TAG2ID


@dataclass
class DetectorOutput:
    sentence_logits: torch.Tensor  # (batch, 2)
    token_logits: torch.Tensor  # (batch, seq_len, num_tags)
    loss: torch.Tensor | None = None
    sentence_loss: torch.Tensor | None = None
    token_loss: torch.Tensor | None = None


class BiasDetector(nn.Module):
    def __init__(
        self,
        backbone: str | None = "launch/POLITICS",
        encoder_config: PretrainedConfig | None = None,
        num_tags: int = len(TAG2ID),
        dropout: float = 0.1,
        sentence_loss_weight: float = 1.0,
        span_loss_weight: float = 1.0,
    ):
        """Build from a pretrained ``backbone`` name (downloads weights — the
        training-time path), or from an already-loaded ``encoder_config``
        (random-init encoder body, immediately overwritten by a fine-tuned
        state dict — the ``from_pretrained`` reload path, see
        ``biasneut.detector.infer.DetectorInference.from_pretrained``)."""
        super().__init__()
        if encoder_config is not None:
            config = encoder_config
            self.encoder = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(backbone)
            self.encoder = AutoModel.from_pretrained(backbone, config=config)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.sentence_head = nn.Linear(hidden_size, 2)
        self.token_head = nn.Linear(hidden_size, num_tags)

        self.sentence_loss_weight = sentence_loss_weight
        self.span_loss_weight = span_loss_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_labels: torch.Tensor | None = None,
        token_labels: torch.Tensor | None = None,
    ) -> DetectorOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(encoded.last_hidden_state)

        pooled = sequence_output[:, 0, :]  # CLS / <s> token
        sentence_logits = self.sentence_head(pooled)
        token_logits = self.token_head(sequence_output)

        loss = sentence_loss = token_loss = None
        if sentence_labels is not None and token_labels is not None:
            sentence_loss = nn.functional.cross_entropy(sentence_logits, sentence_labels)
            token_loss = nn.functional.cross_entropy(
                token_logits.view(-1, token_logits.size(-1)),
                token_labels.view(-1),
                ignore_index=-100,
            )
            loss = self.sentence_loss_weight * sentence_loss + self.span_loss_weight * token_loss

        return DetectorOutput(
            sentence_logits=sentence_logits,
            token_logits=token_logits,
            loss=loss,
            sentence_loss=sentence_loss,
            token_loss=token_loss,
        )
