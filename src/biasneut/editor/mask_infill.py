"""Strategy C, arm 1 (§3.4, §4.2): fully unsupervised Mask-and-Infill.

Scope note vs. the literature this is adapted from: Malmi et al.'s Masker
selects mask positions from *disagreement between two domain-specific MLMs'
token likelihoods*. We already have a stronger, directly-supervised masking
signal from Stage 1 (a BABE/BASIL-trained span detector), so we use that to
choose *where* to mask, and reserve the MLM for *what* to infill — same
delete+infill shape as Masker/Mask-and-Infill, cheaper mask-selection step.
The infiller MLM is optionally domain-adapted by continuing its MLM
objective on human-labeled *neutral* sentences (BABE/BASIL label=0), nudging
its infill distribution toward neutral register — a lightweight stand-in for
Masker's "target-domain MLM."
"""
from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

from biasneut.common.compute import autocast_context, build_optimizer, get_device, resolve_dtype
from biasneut.common.config import ComputeConfig
from biasneut.data.tokenize_utils import whitespace_tokenize

logger = logging.getLogger(__name__)


class MaskInfiller:
    def __init__(self, backbone: str = "roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForMaskedLM.from_pretrained(backbone)
        self.model.eval()

    def to(self, device: torch.device) -> "MaskInfiller":
        self.model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def infill(self, sentence: str, word_tags: list[str], top_k: int = 1) -> str:
        """Replace each flagged word with the MLM's top prediction, filled in
        one joint forward pass (all mask positions predicted simultaneously,
        matching Wu et al. 2019's non-autoregressive infilling)."""
        words = whitespace_tokenize(sentence)
        mask_token = self.tokenizer.mask_token
        masked_words = [mask_token if tag != "O" else w for w, tag in zip(words, word_tags)]
        if mask_token not in masked_words:
            return sentence

        masked_text = " ".join(masked_words)
        encoding = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        logits = self.model(**encoding).logits[0]

        mask_positions = (encoding["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        predicted_ids = encoding["input_ids"][0].clone()
        for pos in mask_positions:
            top_id = logits[pos].topk(top_k).indices[-1]
            predicted_ids[pos] = top_id

        return self.tokenizer.decode(predicted_ids, skip_special_tokens=True)

    def finetune_on_neutral_corpus(
        self,
        neutral_sentences: list[str],
        compute_cfg: ComputeConfig | None = None,
        epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        mlm_probability: float = 0.15,
    ) -> None:
        """Continue the MLM objective on neutral-labeled sentences only, so
        the infiller's predictions lean toward neutral register."""
        compute_cfg = compute_cfg or ComputeConfig()
        device = get_device()
        dtype = resolve_dtype(compute_cfg.mixed_precision)
        self.model.to(device).train()

        encodings = [self.tokenizer(s, truncation=True, max_length=128) for s in neutral_sentences]
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True, mlm_probability=mlm_probability)
        loader = DataLoader(encodings, batch_size=batch_size, shuffle=True, collate_fn=collator)

        optimizer = build_optimizer(self.model, compute_cfg, learning_rate)
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast_context(dtype):
                    out = self.model(**batch)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += out.loss.item()
            logger.info("MLM neutral-domain adaptation epoch %d: loss=%.4f", epoch, total_loss / len(loader))

        self.model.eval()
