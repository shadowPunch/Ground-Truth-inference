"""Detector collation: subword alignment of whitespace-level BIO tags.

Standard NER-style alignment — only the first subword of each whitespace
token carries the tag; continuation subwords and special tokens get -100 so
the token-classification loss ignores them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from biasneut.data.schema import DetectionExample
from biasneut.data.tokenize_utils import whitespace_tokenize

TAG2ID = {"O": 0, "B-BIAS": 1, "I-BIAS": 2}
ID2TAG = {v: k for k, v in TAG2ID.items()}
IGNORE_INDEX = -100


@dataclass
class DetectorCollator:
    tokenizer: object
    max_length: int = 128

    def __call__(self, batch: list[DetectionExample]) -> dict[str, torch.Tensor]:
        word_lists = [whitespace_tokenize(ex.text) for ex in batch]
        encoding = self.tokenizer(
            word_lists,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        token_labels = torch.full_like(encoding["input_ids"], IGNORE_INDEX)
        for i, ex in enumerate(batch):
            word_ids = encoding.word_ids(batch_index=i)
            bio_tags = ex.bio_tags or ["O"] * len(word_lists[i])
            prev_word_id = None
            for j, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != prev_word_id and word_id < len(bio_tags):
                    token_labels[i, j] = TAG2ID[bio_tags[word_id]]
                prev_word_id = word_id

        sentence_labels = torch.tensor([1 if ex.is_biased else 0 for ex in batch], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "sentence_labels": sentence_labels,
            "token_labels": token_labels,
        }
