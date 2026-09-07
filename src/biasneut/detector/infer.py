"""Detector inference wrapper — shared by the editor (span localization) and
the evaluation harness (independent Aggregate Bias Score classifier, §5.1,
§6.1). A *separately trained* instance must be used for evaluation than the
one used anywhere in training/filtering, to keep it independent (§4.3, §6.1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from biasneut.data.collate import ID2TAG
from biasneut.data.tokenize_utils import whitespace_tokenize
from biasneut.detector.model import BiasDetector


@dataclass
class DetectorPrediction:
    text: str
    is_biased: bool
    bias_prob: float
    word_tags: list[str]
    words: list[str]

    @property
    def biased_span_text(self) -> str | None:
        biased_words = [w for w, t in zip(self.words, self.word_tags) if t != "O"]
        return " ".join(biased_words) if biased_words else None


class DetectorInference:
    def __init__(self, model: BiasDetector, tokenizer, device: torch.device | None = None, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_length = max_length
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, backbone: str | None = None) -> "DetectorInference":
        """Reload a checkpoint written by ``biasneut.detector.train.save_pretrained``.

        Rebuilds the encoder architecture from the config.json saved
        alongside the fine-tuned weights (no redownload of the original
        pretrained backbone) unless ``backbone`` is explicitly given, e.g. to
        load an older checkpoint saved before config.json was persisted.
        """
        model_dir = Path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if backbone is not None:
            model = BiasDetector(backbone=backbone)
        else:
            encoder_config = AutoConfig.from_pretrained(model_dir)
            model = BiasDetector(encoder_config=encoder_config)
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return cls(model, tokenizer)

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[DetectorPrediction]:
        word_lists = [whitespace_tokenize(s) for s in sentences]
        encoding = self.tokenizer(
            word_lists, is_split_into_words=True, truncation=True, padding=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)

        out = self.model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        sentence_probs = F.softmax(out.sentence_logits, dim=-1)[:, 1]
        token_preds = out.token_logits.argmax(-1)

        predictions = []
        for i, words in enumerate(word_lists):
            word_ids = encoding.word_ids(batch_index=i)
            word_tags = ["O"] * len(words)
            seen_words = set()
            for j, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen_words or word_id >= len(words):
                    continue
                seen_words.add(word_id)
                word_tags[word_id] = ID2TAG[token_preds[i, j].item()]

            predictions.append(
                DetectorPrediction(
                    text=sentences[i],
                    is_biased=bool(sentence_probs[i].item() > 0.5),
                    bias_prob=sentence_probs[i].item(),
                    word_tags=word_tags,
                    words=words,
                )
            )
        return predictions

    def bias_score(self, sentence: str) -> float:
        """Convenience single-sentence scorer for eval/filtering call sites."""
        return self.predict([sentence])[0].bias_prob
