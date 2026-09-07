# biasneut — lexical political bias detection & neutralization

Implementation of `political_bias_neutralization_proposal.md` (see the repo
root): a **detect-then-edit** system for sentence-level lexical political
bias in news text, trainable on free-tier compute. This folder is the actual
codebase; the proposal is the design document it implements.

Nothing in `bias_data/`, `neutralizing-bias-master/`, `program/`,
`submission/`, or `test/` at the repo root is reused as running code — that
existing code targets the *concurrent* model the proposal explicitly drops,
or is unfinished/mock scaffolding (broken imports, missing vocab files, a
`time.sleep()`-based fake backend). This is a from-scratch implementation of
the proposal's own architecture. Two assets *are* reused as data: the
Wikipedia Neutrality Corpus under `bias_data/bias_data/WNC/` (Strategy A) and
the 14 Recasens-style lexicon files under
`neutralizing-bias-master/src/lexicons/` (Aggregate Bias Score).

## Architecture

```
STAGE 1 — Detector (biasneut.detector)         STAGE 2 — Editor (biasneut.editor)
  RoBERTa-scale encoder + 2 heads:                One of three strategies (§4.2):
   • sentence head: biased / neutral               A. WNC pretrain (+ optional adapt)
   • token head: BIO tag of biased span(s)          B. LLM-synthesized pseudo-parallel
  trained on BABE (sentence) + BASIL (span,          C. unsupervised mask-and-infill /
  lexical-only — see §2.1 scope note below)             LEWIS-lite
        │ predicted span                            All three fine-tune the *same*
        └──────────────────────────────────────►    T5-small/BART-base seq2seq editor
                                                      over span-marked input, with an
                                                      optional decode-time copy bias.

Both stages are wired together by biasneut.pipeline.BiasNeutralizationPipeline.
Evaluation (biasneut.eval) scores transfer strength × preservation × fluency
jointly, against mandatory trivial baselines (biasneut.baselines), with
Pareto plots and significance testing.
```

### Package layout

```
src/biasneut/
  common/     config (dataclass+YAML), checkpoint/resume, compute (precision/
              grad-checkpointing/8-bit optim), seeding, logging
  data/       BABE/BASIL/WNC loaders, leakage-safe splitting, pseudo-parallel
              generation (Strategy B) + §4.3 circularity mitigations
  detector/   dual-head model, training loop, inference wrapper, span metrics
  editor/     span marking, copy-bias decoding, the shared seq2seq model, one
              training module per strategy (a/b/c), QLoRA stretch arm
  baselines/  copy-input, delete-flagged-word, Pryzant-off-the-shelf stand-in,
              LLM zero-shot
  eval/       lexicon features, transfer strength, preservation, fluency,
              significance testing, Pareto frontier, the full harness
  pipeline.py end-to-end detect-then-edit inference
scripts/      one CLI per pipeline stage (see "Running it" below)
configs/      example YAML configs for the detector and each editor strategy
tests/        unit tests (fast, no network) + a smaller set of `@pytest.mark
              .integration` tests that pull small real HF models/metrics
```

## Setup

```bash
cd implement
uv venv --python 3.12 .venv        # transformers/peft/bitsandbytes lag on
                                     # bleeding-edge Python; 3.12 is the safe choice
uv pip install -e ".[dev]"          # add "qlora" for the stretch decoder arm,
                                     # "llm" for real (non-echo) Strategy-B generation
```

Verified in this environment: `torch 2.12.1+cu130` with CUDA available,
`transformers 5.13.0`.

## Running it

```bash
# Phase 0 — acquire data + build leakage-safe splits (§6.5: no story/outlet
# straddles a split boundary). BASIL is shallow-git-cloned on first use;
# BABE comes from the HF Hub; WNC is read from the local bias_data/ release.
python scripts/prepare_data.py --cache-dir data_cache

# Phase 1 — Stage-1 detector. Run twice with different --set output_dir=...
# to get a pipeline instance and a separate *independent* eval instance
# (§4.3/§5.1/§6.1 — the eval instance must never see training-time filtering).
python scripts/train_detector.py --cache-dir data_cache --config configs/detector.yaml
python scripts/train_detector.py --cache-dir data_cache --config configs/detector.yaml \
    --set output_dir=runs/detector_eval_independent --seed 1337

# Phase 2 — Stage-2 editor, one arm at a time (§8: these can run in any order
# / in parallel across the weekly compute quota).
python scripts/train_editor.py --strategy a --cache-dir data_cache --config configs/editor_strategy_a.yaml

python scripts/generate_pseudo_parallel.py --cache-dir data_cache \
    --detector-dir runs/detector_eval_independent --dry-run   # --dry-run = EchoClient, no API key needed
python scripts/train_editor.py --strategy b --config configs/editor_strategy_b.yaml \
    --pseudo-parallel-path runs/pseudo_parallel_filtered.json

python scripts/train_editor.py --strategy c --arm lewis --config configs/editor_strategy_c.yaml \
    --detector-dir runs/detector

# Inference
python scripts/run_pipeline.py --detector-dir runs/detector --editor-dir runs/editor_strategy_a \
    --sentence "The corrupt regime brutally cracked down on peaceful protesters."

# Phase 5 — full evaluation triad + baselines + Pareto + significance (§6)
python scripts/run_evaluation.py --cache-dir data_cache \
    --pipeline-detector-dir runs/detector --independent-detector-dir runs/detector_eval_independent \
    --editor-dir runs/editor_strategy_a
```

`--set key=value` (repeatable) overrides any config field on the CLI,
including nested ones via dotted keys, e.g. `--set compute.mixed_precision=fp16`
(use `fp16` on a P100 — no tensor cores, §7.2 point 3) or
`compute.gradient_checkpointing=false`.

## Design decisions worth knowing before extending this

These are the places where turning the proposal into runnable code required a
concrete choice the proposal itself leaves open, or where being honest about
scope meant not building something:

- **Pryzant off-the-shelf baseline.** The original 2019 release pins
  `pytorch_pretrained_bert==0.3.0` + `torch==1.1.0` and is not compatible with
  a modern stack. Rather than resurrect that environment, `scripts/train_editor.py
  --strategy a` (no `adapt_train`) *is* the stand-in: it answers the same
  question ("how far does pure Wikipedia transfer get on news?") with the
  same modern architecture as everything else here, so the comparison stays
  apples-to-apples. See `biasneut/baselines/trivial.py`'s module docstring.
- **Constrained decoding is a soft copy bias, not a pointer-generator.**
  Pryzant's copy-heavy inductive bias comes from a *trained* pointer-generator
  decoder tied to his custom LSTM decoder, which we deliberately didn't port
  forward. `biasneut/editor/constrained_decoding.py` instead boosts the
  logits of subword tokens that occur in the sentence's non-flagged region at
  every decoding step — a position-agnostic nudge toward reusing source
  vocabulary, not a hard copy constraint. Documented in that module's
  docstring so it isn't mistaken for a faithful port.
- **Strategy C's "Masker" arm uses detector-guided masking, not dual-MLM
  disagreement.** Malmi et al.'s Masker selects mask positions from where two
  domain MLMs disagree most. We already have a stronger, directly-supervised
  masking signal — Stage 1's trained span detector — so `MaskInfiller` uses
  that instead, and reserves the MLM purely for infilling (optionally
  domain-adapted toward neutral register via continued MLM training on
  BABE/BASIL neutral-labeled sentences).
- **LEWIS-lite reuses the Stage-1 detector as the tagger.** Full LEWIS trains
  a second bespoke RoBERTa insert/replace/delete tagger. `biasneut/editor/lewis.py`
  reuses the already-trained Stage-1 detector for masking decisions instead,
  and trains the *same* shared seq2seq editor on the resulting synthesized
  pairs — a working LEWIS-lite arm without a second architecture to maintain.
- **BASIL loads only lexical spans as detection targets** (§2.1's scope
  decision is load-bearing, not incidental): informational-bias phrase
  annotations are read but discarded rather than folded into the same BIO
  tags, since Pryzant's copy-heavy editor is the wrong tool for that half of
  the problem by the proposal's own argument.
- **`launch/POLITICS` is roberta-**base** scale** (125M params, 12 layers,
  768 hidden — confirmed from its published `config.json`), not
  roberta-large scale as sometimes described. The code reads config
  dynamically so this doesn't affect correctness, just the "355M" figure
  that shows up in some framings of this backbone.
- **QLoRA stretch arm is optional and isolated.** `biasneut/editor/qlora_decoder.py`
  lazy-imports `bitsandbytes`/`peft`'s k-bit path and is gated behind the
  `qlora` extra; nothing else in the codebase depends on it (§7.3: "none of
  the core pipeline requires QLoRA").

## What's been verified vs. what's scaffolded

Verified against real data/models during development (not just unit-tested
against synthetic fixtures):

- **BABE** loads correctly from the HF Hub (`mediabiasgroup/BABE`, 3121
  sentences) with correct BIO tagging of its `biased_words` field.
- **BASIL** loads correctly via shallow git clone of `launchnlp/BASIL` — 300
  article/annotation pairs → 7984 sentences (matches the paper's own
  reported corpus size exactly), with correct lexical-span BIO tagging and
  story-disjoint grouping via `triplet-uuid` (100 unique stories × 3 outlets).
  If `git clone` fails (observed in one network-restricted sandbox: HTTPS to
  `github.com` itself blocked while `codeload.github.com` stayed reachable),
  `ensure_basil_repo` falls back to downloading + extracting the same repo as
  a tarball via stdlib `urllib`/`tarfile` — verified to produce identical
  output (same 7984 sentences).
- **WNC** loads correctly from the locally-vendored TSV release.
- Leakage-safe splitting produces no cross-split story leakage on real BASIL
  data (verified via `assert_no_leakage`).
- The detector CLI (`train_detector.py`) runs a real training loop
  end-to-end on real BABE/BASIL data with a real HF backbone
  (`distilroberta-base`), including checkpoint save **and resume** (verified:
  a second invocation with a higher `num_epochs` correctly resumed from the
  saved step instead of restarting).
- A full (non-toy) 5-epoch run on the complete real BABE+BASIL train split
  (8888 examples, `distilroberta-base`, a single 4GB GPU, ~5 minutes) shows
  the token/span head actually learning: span F1 goes 0.0 → 0.053 → 0.27 →
  ~0.22-0.23 across epochs, after starting from an all-"O" prediction at
  epoch 0. (A smaller smoke run — 500 examples, 4 epochs — never got span F1
  off 0.0; that turned out to be a class-imbalance/scale artifact, not a bug:
  only ~3.5% of BABE tokens are inside a biased span, so an unweighted
  cross-entropy token head needs real data volume to escape predicting the
  majority class everywhere. Confirmed by inspection of the loss computation
  in `detector/model.py` and the subword-label alignment in
  `data/collate.py` — both are correct. Worth watching on the real Kaggle
  run: if span F1 is still ~0 after a few real epochs on the *full*
  BABE+BASIL set with the real `launch/POLITICS` backbone, add class
  weighting to the `cross_entropy` call for `token_labels`.)
- A real Strategy-A editor (`t5-small`) trained on a real WNC subset, saved,
  reloaded, and driven through `scripts/run_pipeline.py` end-to-end.
- Strategy B (`generate_pseudo_parallel.py --dry-run` → `train_editor.py
  --strategy b`) and Strategy C's `mask_infill` arm both run end-to-end.
  Strategy C's `lewis` arm needs a detector with genuine span recall — with
  the from-cold-start smoke detector above (all-"O" predictions) it
  synthesized 0 pairs, which used to crash `DataLoader` with an opaque
  `num_samples=0` error (`train_seq2seq_editor` now raises a clear message
  instead, see below); re-run against the real-scale detector (span F1 ~0.23
  above), it synthesized 609 pseudo-parallel pairs from 1424 candidate
  sentences and trained the shared seq2seq editor on them successfully.
- The full evaluation harness (`EvaluationHarness` + `run_full_evaluation`)
  ran end-to-end against real components (SBERT, BERTScore, SARI, BLEU, GPT-2
  perplexity, CoLA grammaticality, the Pryzant lexicon files) and produced a
  correctly-formed report, significance tests, and Pareto plot.

None of this constitutes a real training run at the scale the proposal's
16-week work plan describes (§8) — that's real GPU-time the user needs to
spend on Kaggle, not something to fake here. What's verified is that the
*pipeline itself* is correct: every stage runs, checkpoints, resumes, and
hands off to the next stage without silent shape/API mismatches. Several real
bugs were caught and fixed this way (not found by unit tests against
synthetic data): a nested-dataclass config round-trip that silently produced
unusable objects, a missing `sacremoses` runtime dependency for the SARI
metric, a transformers-5.x tokenizer API change, `DetectorInference.
from_pretrained` crashing because the training loop never persisted the
encoder's `config.json` (so no saved detector checkpoint could actually be
reloaded), `configs/*.yaml` writing learning rates as `2e-5`/`3e-4` — PyYAML's
resolver silently parses scientific notation without a decimal point as a
*string*, not a float, which crashed `torch.optim.AdamW` deep inside training
instead of failing at config-load time (fixed in the YAML files and hardened
in `common/config.py`'s loader so it can't recur), and the opaque
zero-training-examples `DataLoader` crash in Strategy C/LEWIS noted above.

Fast unit tests (`pytest -m "not integration"`, 90 tests, no network) cover
the pure logic: BIO tagging and alignment, span marking, leakage-safe
splitting, detector/editor metrics, significance testing, Pareto frontiers,
checkpoint save/load, and config (de)serialization. A second tier (6 tests,
`@pytest.mark.integration`) exercises the same logic against small real HF
models and real data sources (BASIL's git-clone/tarball-fallback path
included) to catch API-version drift and wiring bugs that synthetic mocks
can't.

## Data governance (§4.3)

Synthetic (Strategy B) data is training-side only and must never enter the
BABE/BASIL-derived test split — `filter_pseudo_parallel` enforces the
bias-drop and similarity-floor mitigations, and `sample_for_human_review`
exports a random subset for the mandatory human spot-check. Confirm
redistribution licenses for any dataset before publishing derived artifacts;
BABE/BASIL are both used here under their respective research licenses via
the HF Hub / original GitHub release.
