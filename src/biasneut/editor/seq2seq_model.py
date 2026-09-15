"""Stage-2 primary editor (§5.2): T5-small/BART-base seq2seq over
span-marked input, with optional decode-time copy bias (§5.2, §7.3 — small
enough to fully fine-tune on a free GPU, no PEFT needed for this arm)."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from biasneut.editor.constrained_decoding import build_copy_bias_processor
from biasneut.editor.span_marking import mark_text


class SeqEditor:
    def __init__(self, model, tokenizer, task_prefix: str = ""):
        self.model = model
        self.tokenizer = tokenizer
        self.task_prefix = task_prefix

    @classmethod
    def from_pretrained_backbone(cls, backbone: str = "t5-small") -> "SeqEditor":
        tokenizer = AutoTokenizer.from_pretrained(backbone)
        model = AutoModelForSeq2SeqLM.from_pretrained(backbone)
        task_prefix = "neutralize: " if backbone.startswith("t5") else ""
        editor = cls(model, tokenizer, task_prefix=task_prefix)
        editor._add_span_markers()
        return editor

    @classmethod
    def load(cls, model_dir: str | Path) -> "SeqEditor":
        model_dir = Path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        task_prefix = (model_dir / "task_prefix.txt").read_text() if (model_dir / "task_prefix.txt").exists() else ""
        return cls(model, tokenizer, task_prefix=task_prefix)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        (model_dir / "task_prefix.txt").write_text(self.task_prefix)

    def _add_span_markers(self, span_start: str = "<bias>", span_end: str = "</bias>") -> None:
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": [span_start, span_end]})
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def to(self, device: torch.device) -> "SeqEditor":
        self.model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def neutralize(
        self,
        sentences: list[str],
        word_tags_per_sentence: list[list[str]],
        num_beams: int = 4,
        max_new_tokens: int = 128,
        use_constrained_decoding: bool = True,
        copy_bias_strength: float = 2.5,
        no_repeat_ngram_size: int = 3,
    ) -> list[str]:
        marked = [
            self.task_prefix + mark_text(s, tags)
            for s, tags in zip(sentences, word_tags_per_sentence)
        ]
        inputs = self.tokenizer(marked, return_tensors="pt", padding=True, truncation=True, max_length=256).to(
            self.device
        )

        logits_processor = None
        if use_constrained_decoding:
            logits_processor = build_copy_bias_processor(
                self.tokenizer, sentences, word_tags_per_sentence, bias_strength=copy_bias_strength
            )

        # The copy-bias boost above pushes up every non-flagged token uniformly
        # at every step, which without a repetition guard lets beam search loop
        # on whichever boosted token scores highest (observed: real outputs
        # collapsing to "protest protest protest..."). no_repeat_ngram_size is
        # the standard HF safeguard against exactly this.
        output_ids = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
