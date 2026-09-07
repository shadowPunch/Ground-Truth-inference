"""Mandatory honesty-check baselines (§6.3) — every one of these is annoyingly
easy to beat trivially and surprisingly hard to beat for real, which is
exactly why the proposal insists on reporting them.

Note on "Pryzant off-the-shelf (WNC-trained)": the original 2019 release
(``neutralizing-bias-master/``) is pinned to ``pytorch_pretrained_bert==0.3.0``
and ``torch==1.1.0`` and is not load-bearing-compatible with a modern stack
(see project survey). Rather than resurrect that environment, we substitute
our own Strategy-A editor *after WNC pretraining but before any news
adaptation* — i.e. ``train_strategy_a.run_strategy_a(..., adapt_train=None)``
— which answers the exact same question the baseline is meant to answer
("how far does pure Wikipedia transfer get on news?") using the same modern
architecture as the rest of this codebase, so the comparison stays apples to
apples with our own ablation arms.
"""
from __future__ import annotations

from biasneut.data.pseudo_parallel import NEUTRALIZE_PROMPT_TEMPLATE, LLMClient
from biasneut.data.tokenize_utils import whitespace_tokenize
from biasneut.detector.infer import DetectorInference
from biasneut.editor.seq2seq_model import SeqEditor


def copy_input_baseline(sentences: list[str]) -> list[str]:
    """Upper bound on preservation, zero transfer strength by construction."""
    return list(sentences)


def delete_flagged_word_baseline(sentences: list[str], word_tags_per_sentence: list[list[str]]) -> list[str]:
    """Pryzant's trivial editor: delete every token the detector flagged,
    keep everything else untouched."""
    outputs = []
    for sentence, tags in zip(sentences, word_tags_per_sentence):
        words = whitespace_tokenize(sentence)
        kept = [w for w, t in zip(words, tags) if t == "O"]
        outputs.append(" ".join(kept))
    return outputs


def delete_flagged_word_with_detector(sentences: list[str], detector: DetectorInference) -> list[str]:
    predictions = detector.predict(sentences)
    return delete_flagged_word_baseline(sentences, [p.word_tags for p in predictions])


def pryzant_offshelf_baseline(sentences: list[str], word_tags_per_sentence: list[list[str]],
                               wnc_pretrained_editor: SeqEditor, num_beams: int = 4) -> list[str]:
    """See module docstring: this *is* the WNC-pretrain-only Strategy-A
    checkpoint, used as the stand-in for Pryzant's original release."""
    return wnc_pretrained_editor.neutralize(
        sentences, word_tags_per_sentence, num_beams=num_beams, use_constrained_decoding=False,
    )


def llm_zero_shot_baseline(sentences: list[str], client: LLMClient,
                            prompt_template: str = NEUTRALIZE_PROMPT_TEMPLATE) -> list[str]:
    """Strong-but-not-free reference ceiling — contextualizes what a small,
    free-GPU system gives up (§6.3)."""
    return [client.generate(prompt_template.format(sentence=s)) for s in sentences]
