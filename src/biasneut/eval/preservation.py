"""Content preservation (§6.1, axis 2): BERTScore + SBERT cosine + SARI + BLEU.

"Preservation alone rewards the copy-input degenerate solution, so it is
never read in isolation" — this module only computes the numbers; joint
interpretation happens in ``eval.harness``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import evaluate
import sacrebleu
from sentence_transformers import SentenceTransformer, util


@dataclass
class PreservationResult:
    bertscore_f1: list[float]
    sbert_cosine: list[float]
    sari: float
    bleu: float


class PreservationScorer:
    def __init__(self, sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 bertscore_model: str = "roberta-large"):
        self._sbert = SentenceTransformer(sbert_model)
        self._bertscore_model = bertscore_model
        self._bertscore = evaluate.load("bertscore")
        self._sari = evaluate.load("sari")

    def score(self, sources: list[str], outputs: list[str], references: list[str] | None = None) -> PreservationResult:
        """``references`` are human-neutralized targets when available (e.g.
        held-out WNC pairs or a human-annotated subset); for arms without a
        gold target, pass ``sources`` again so SARI/BLEU degrade gracefully
        to measuring similarity against the input instead of a gold rewrite.
        """
        references = references or sources

        bert_out = self._bertscore.compute(
            predictions=outputs, references=references, model_type=self._bertscore_model
        )

        src_emb = self._sbert.encode(sources, convert_to_tensor=True)
        out_emb = self._sbert.encode(outputs, convert_to_tensor=True)
        sbert_cosine = util.pairwise_cos_sim(src_emb, out_emb).tolist()

        sari_result = self._sari.compute(
            sources=sources, predictions=outputs, references=[[r] for r in references]
        )
        bleu = sacrebleu.corpus_bleu(outputs, [references]).score

        return PreservationResult(
            bertscore_f1=bert_out["f1"],
            sbert_cosine=sbert_cosine,
            sari=sari_result["sari"],
            bleu=bleu,
        )
