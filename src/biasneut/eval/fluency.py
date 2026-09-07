"""Fluency (§6.1, axis 3): perplexity under a fixed external LM + CoLA
grammaticality. "The guardrail that stops the model from 'neutralizing' by
mangling the sentence." The LM must stay fixed across all systems compared
(never the model being evaluated) so scores are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from biasneut.common.compute import get_device


@dataclass
class FluencyResult:
    perplexities: list[float]
    grammatical_probs: list[float]


class FluencyScorer:
    def __init__(self, lm_name: str = "gpt2", grammaticality_model: str = "textattack/roberta-base-CoLA",
                 device: torch.device | None = None):
        self.device = device or get_device()

        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name).to(self.device).eval()

        self.cola_tokenizer = AutoTokenizer.from_pretrained(grammaticality_model)
        self.cola_model = AutoModelForSequenceClassification.from_pretrained(grammaticality_model).to(
            self.device
        ).eval()

    @torch.no_grad()
    def perplexity(self, sentence: str) -> float:
        if not sentence.strip():
            return float("inf")
        encoding = self.lm_tokenizer(sentence, return_tensors="pt").to(self.device)
        # A single-token sequence has no next-token position to score against,
        # so the causal-LM loss is NaN (observed from a degenerate one-token
        # editor output on real data) — treat it the same as empty: maximally
        # non-fluent, not a silent NaN that poisons the aggregate mean.
        if encoding["input_ids"].shape[1] < 2:
            return float("inf")
        out = self.lm(**encoding, labels=encoding["input_ids"])
        return torch.exp(out.loss).item()

    @torch.no_grad()
    def grammatical_prob(self, sentence: str) -> float:
        encoding = self.cola_tokenizer(sentence, return_tensors="pt", truncation=True).to(self.device)
        logits = self.cola_model(**encoding).logits
        probs = F.softmax(logits, dim=-1)[0]
        # textattack/roberta-base-CoLA: label 1 = acceptable/grammatical.
        return probs[1].item()

    def score(self, sentences: list[str]) -> FluencyResult:
        return FluencyResult(
            perplexities=[self.perplexity(s) for s in sentences],
            grammatical_probs=[self.grammatical_prob(s) for s in sentences],
        )
