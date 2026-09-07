"""Strategy B — LLM-synthesized pseudo-parallel corpus (§4.2, §4.3).

Generation happens offline (no GPU cost) against one or more LLM providers;
the resulting pairs are training-side only and must never leak into
evaluation (§4.3: "Synthetic data never touches evaluation"). This module
implements generation plus the three mandatory circularity mitigations:

1. multi-model generation (``clients`` accepts >=1 distinct model families)
2. independent-classifier filtering (reject targets that don't reduce bias)
3. semantic-similarity floor (reject targets that drift from the source)

Human spot-check (the fourth mitigation) is inherently manual; ``sample_for_
human_review`` exports a random subset for that step rather than automating it.
"""
from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from biasneut.data.schema import DetectionExample, EditExample

logger = logging.getLogger(__name__)

NEUTRALIZE_PROMPT_TEMPLATE = """You are helping build a training set for a lexical bias neutralization model.

Rewrite the sentence below to remove loaded/partisan word choice (e.g. \
epithets, slanted verbs, framing adjectives) while:
- preserving every factual claim, entity, and quote exactly
- changing as few words as possible (this is a local edit task, not a rewrite)
- keeping the same sentence structure and length where possible

If the sentence is already neutral, return it unchanged.
Return ONLY the rewritten sentence, no explanation, no quotes.

Sentence: {sentence}"""


class LLMClient(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


@dataclass
class AnthropicClient:
    """Thin wrapper so the generation loop doesn't depend on the SDK shape."""

    model: str = "claude-sonnet-5"
    name: str = "anthropic"
    max_tokens: int = 256

    def __post_init__(self):
        import anthropic

        self._client = anthropic.Anthropic()

    def generate(self, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


@dataclass
class EchoClient:
    """No-op client for tests/dry-runs: returns the input unchanged so the
    generation/filtering pipeline can be exercised without any API key."""

    name: str = "echo"

    def generate(self, prompt: str) -> str:
        # Extract the sentence back out of the templated prompt.
        marker = "Sentence: "
        idx = prompt.rfind(marker)
        return prompt[idx + len(marker):].strip() if idx != -1 else prompt


def generate_pseudo_parallel(
    examples: list[DetectionExample],
    clients: list[LLMClient],
    prompt_template: str = NEUTRALIZE_PROMPT_TEMPLATE,
) -> list[EditExample]:
    """One candidate per (example, client) — deliberately not deduplicated
    here so downstream filtering can compare candidates from different model
    families against the same source sentence."""
    pairs: list[EditExample] = []
    for ex in examples:
        if not ex.is_biased:
            continue
        prompt = prompt_template.format(sentence=ex.text)
        for client in clients:
            try:
                target = client.generate(prompt)
            except Exception:
                logger.exception("Generation failed for client=%s on: %s", client.name, ex.text[:80])
                continue
            pairs.append(
                EditExample(
                    source=ex.text,
                    target=target,
                    biased_span=None,
                    strategy="llm_synth",
                    provenance=f"model={client.name}",
                )
            )
    return pairs


def filter_pseudo_parallel(
    pairs: list[EditExample],
    bias_score_fn: Callable[[str], float],
    similarity_fn: Callable[[str, str], float],
    similarity_floor: float = 0.6,
    require_bias_drop: bool = True,
) -> list[EditExample]:
    """Apply §4.3's circularity mitigations.

    ``bias_score_fn`` and ``similarity_fn`` must come from the *independent*
    evaluator (a BABE-trained classifier instance that never saw generation
    outputs, and an SBERT model) — never from the generating LLM itself.
    """
    kept = []
    for pair in pairs:
        sim = similarity_fn(pair.source, pair.target)
        if sim < similarity_floor:
            continue
        if require_bias_drop:
            source_score = bias_score_fn(pair.source)
            target_score = bias_score_fn(pair.target)
            if target_score >= source_score:
                continue
        kept.append(pair)
    logger.info("Filtered pseudo-parallel pairs: %d -> %d", len(pairs), len(kept))
    return kept


def sample_for_human_review(pairs: list[EditExample], n: int = 200, seed: int = 42,
                             out_path: str | Path = "human_review_sample.csv") -> Path:
    sample = random.Random(seed).sample(pairs, k=min(n, len(pairs)))
    out_path = Path(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "provenance", "bias_removed_ok", "meaning_preserved_ok", "fluent_ok"])
        for pair in sample:
            writer.writerow([pair.source, pair.target, pair.provenance, "", "", ""])
    return out_path
