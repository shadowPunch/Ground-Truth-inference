"""Editor training collation: mark the diff-derived biased span in the
source, tokenize (source, target) pairs for seq2seq training."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from biasneut.data.schema import EditExample
from biasneut.editor.span_marking import diff_word_tags, mark_text


@dataclass
class EditorCollator:
    tokenizer: object
    max_source_length: int = 128
    max_target_length: int = 128
    task_prefix: str = ""

    def __call__(self, batch: list[EditExample]) -> dict[str, torch.Tensor]:
        marked_sources = []
        for ex in batch:
            tags = diff_word_tags(ex.source, ex.target)
            marked_sources.append(self.task_prefix + mark_text(ex.source, tags))

        model_inputs = self.tokenizer(
            marked_sources, truncation=True, padding=True, max_length=self.max_source_length, return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=[ex.target for ex in batch],
            truncation=True, padding=True, max_length=self.max_target_length, return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs
