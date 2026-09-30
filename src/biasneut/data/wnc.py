"""WNC loader (§3.1, §4.1, Strategy A) — Wikipedia Neutrality Corpus, used
*only* for editor pretraining, never as primary supervision (the whole
premise of this project is that WNC's Wikipedia register does not transfer
cleanly to news, per §2.1/§4.2).

File format (the official release, https://nlp.stanford.edu/projects/bias/bias_data.zip):
7-column TSV, no header —
``id, src_tok, tgt_tok, src_raw, tgt_raw, src_pos_tags, tgt_dep_tags``.
We use only the raw (untokenized) columns; the BERT-wordpiece columns are
Pryzant-specific artifacts we don't need with a modern tokenizer.
"""
from __future__ import annotations

import csv
import difflib
from pathlib import Path

from biasneut.data.schema import EditExample

_COLUMNS = ("id", "src_tok", "tgt_tok", "src_raw", "tgt_raw", "src_pos", "tgt_dep")


def _extract_removed_span(src_raw: str, tgt_raw: str) -> str | None:
    """Best-effort extraction of the word(s) removed/changed going src->tgt,
    for reporting/inspection only (training uses the full src/tgt pair)."""
    sm = difflib.SequenceMatcher(a=src_raw.split(), b=tgt_raw.split())
    removed = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed.extend(src_raw.split()[i1:i2])
    return " ".join(removed) if removed else None


def load_wnc(path: str | Path, max_examples: int | None = None) -> list[EditExample]:
    path = Path(path)
    examples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            row_dict = dict(zip(_COLUMNS, row))
            src_raw, tgt_raw = row_dict["src_raw"], row_dict["tgt_raw"]
            examples.append(
                EditExample(
                    source=src_raw,
                    target=tgt_raw,
                    biased_span=_extract_removed_span(src_raw, tgt_raw),
                    strategy="wnc",
                    provenance=f"WNC id={row_dict['id']}",
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples
