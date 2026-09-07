"""Copy-biased decoding (§5.2): "bias the decoder toward copying non-flagged
tokens ... the operational core of Pryzant's inductive bias."

Honest scope note: Pryzant's own copy-heavy inductive bias comes from a
*trained* pointer-generator decoder (position-aware copy distribution),
which is architecturally tied to his custom LSTM decoder (`seq2seq/model.py`
`PointerSeq2Seq`) — we deliberately did not port that decoder forward
(§3.1: drop the concurrent model's machinery). Reimplementing a true
pointer-generator on top of a modern HF T5/BART decoder would require
surgery on the decoder's output projection, not just a logits processor.

What we implement instead is a decode-time *vocabulary bias*: at every
generation step, boost the logits of subword tokens that occur in the
**non-flagged** portion of the source sentence, so the model is nudged
toward reusing exact source vocabulary for content it wasn't told to edit.
This is a softer, position-agnostic approximation of the same inductive
bias — not a hard copy constraint — and is reported as such rather than
oversold as a faithful pointer-generator port.
"""
from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from biasneut.data.tokenize_utils import whitespace_tokenize


class CopyBiasLogitsProcessor(LogitsProcessor):
    """Boosts source-vocabulary logits, one allowed set per *source sentence*.

    ``generate()`` expands the batch to ``num_sources * num_beams`` rows
    (``repeat_interleave``, so row ``r`` belongs to source ``r // num_beams``).
    Indexing ``scores`` by source position instead would bias beam ``k`` of
    sentence ``i`` toward sentence ``i + k``'s vocabulary and leave every row
    past ``num_sources`` untouched — i.e. actively reward copying words from
    a *different* sentence.
    """

    def __init__(self, allowed_token_ids_per_batch: list[set[int]], bias_strength: float):
        self.allowed_token_ids_per_batch = allowed_token_ids_per_batch
        self.bias_strength = bias_strength
        self._index_cache: dict[int, torch.Tensor] = {}

    def _indices(self, source_idx: int, device: torch.device) -> torch.Tensor:
        """Cached id tensor per source — this runs at every decoding step."""
        cached = self._index_cache.get(source_idx)
        if cached is None or cached.device != device:
            allowed = self.allowed_token_ids_per_batch[source_idx]
            cached = torch.tensor(sorted(allowed), device=device, dtype=torch.long)
            self._index_cache[source_idx] = cached
        return cached

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_sources = len(self.allowed_token_ids_per_batch)
        if num_sources == 0:
            return scores

        num_rows = scores.shape[0]
        beams_per_source = max(1, num_rows // num_sources)
        for row in range(num_rows):
            source_idx = min(row // beams_per_source, num_sources - 1)
            if not self.allowed_token_ids_per_batch[source_idx]:
                continue
            idx = self._indices(source_idx, scores.device)
            scores[row, idx] = scores[row, idx] + self.bias_strength
        return scores


def non_flagged_token_ids(tokenizer, source_text: str, word_tags: list[str]) -> set[int]:
    """Subword vocab ids for words *outside* the flagged span in one source
    sentence (i.e. content the model should be biased toward copying)."""
    words = whitespace_tokenize(source_text)
    non_flagged_words = [w for w, t in zip(words, word_tags) if t == "O"]
    if not non_flagged_words:
        return set()
    ids = tokenizer(" ".join(non_flagged_words), add_special_tokens=False)["input_ids"]
    return set(ids)


def build_copy_bias_processor(
    tokenizer,
    sources: list[str],
    word_tags_per_source: list[list[str]],
    bias_strength: float = 2.5,
) -> LogitsProcessorList:
    allowed = [non_flagged_token_ids(tokenizer, src, tags) for src, tags in zip(sources, word_tags_per_source)]
    return LogitsProcessorList([CopyBiasLogitsProcessor(allowed, bias_strength)])
