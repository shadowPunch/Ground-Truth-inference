# Political Bias Detection & Neutralization in News

This repository implements a **detect-then-edit** system for sentence-level
lexical political bias in news text. A detector identifies biased sentences
and the words responsible, and a sequence-to-sequence editor rewrites the
flagged span into neutral language. The system adapts the modular
localize-then-edit design of Pryzant et al. (2020) to the news domain and
trains end to end on a single free-tier GPU. It is evaluated jointly on bias
reduction, meaning preservation and fluency against trivial baselines, with
significance testing. The design document is [`docs/proposal.md`](docs/proposal.md).

## Architecture

```
STAGE 1 — Detector                              STAGE 2 — Editor
  RoBERTa-scale encoder, two heads:               t5-small over span-marked input
   • sentence: biased / neutral                   (<bias> … </bias>) with decode-time
   • token: BIO tags of biased spans              copy bias, trained under three strategies:
  trained on BABE + BASIL                          A. WNC transfer (+ in-domain adaptation)
        │ predicted span                           B. LLM-synthesized pseudo-parallel pairs
        └──────────────────────────────────►       C. unsupervised mask-and-infill (LEWIS-lite)
```

Sentences that the detector does not flag pass through unchanged. Bias in the
outputs is scored by an independent detector instance that is never used
during training.

## Repository structure

```
src/biasneut/    package: data, detector, editor, baselines, eval, pipeline, common
scripts/         command-line entry point for each stage
configs/         YAML configurations
notebooks/       kaggle_pipeline.ipynb — the complete experiment
data/lexicons/   bias lexicons (Recasens et al., 2013; MIT, via Pryzant et al.)
results/         final evaluation outputs
docs/            project proposal
tests/           unit and integration tests
```

## Usage

**Requirements:** Python 3.12, a CUDA GPU (a single T4 is sufficient), a
Weights & Biases API key (`WANDB_API_KEY`), and the WNC corpus. BABE and BASIL
are downloaded automatically.

```bash
uv venv --python 3.12 .venv && uv pip install -e .
mkdir -p data && curl -L https://nlp.stanford.edu/projects/bias/bias_data.zip -o data/bias_data.zip \
  && unzip -q data/bias_data.zip -d data && rm data/bias_data.zip

python scripts/prepare_data.py --cache-dir data_cache
python scripts/train_detector.py --cache-dir data_cache --config configs/detector.yaml
python scripts/train_detector.py --cache-dir data_cache --config configs/detector.yaml \
    --set output_dir=runs/detector_eval_independent --seed 1337
python scripts/train_editor.py --strategy a --cache-dir data_cache --config configs/editor_strategy_a.yaml
python scripts/train_editor.py --strategy c --arm lewis --config configs/editor_strategy_c.yaml \
    --detector-dir runs/detector
python scripts/run_evaluation.py --cache-dir data_cache --pipeline-detector-dir runs/detector \
    --independent-detector-dir runs/detector_eval_independent --editor-dir runs/editor_strategy_a
python scripts/run_pipeline.py --detector-dir runs/detector --editor-dir runs/editor_strategy_a \
    --sentence "The corrupt regime brutally cracked down on peaceful protesters."
```

Alternatively, run [`notebooks/kaggle_pipeline.ipynb`](notebooks/kaggle_pipeline.ipynb)
on Kaggle with Internet enabled, a T4 GPU and a `WANDB_API_KEY` secret. It
executes the complete experiment in about 2.5 hours. Strategy B additionally
requires a `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`.

## Experiment tracking

All training, evaluation and inference runs are logged to Weights & Biases
(project `biasneut`): configurations, training curves, evaluation tables and
outputs. A run stops with an error if credentials are missing. Setting
`BIASNEUT_WANDB=0` disables tracking.

## Results

Evaluated on the held-out BABE test split. Results were consistent across
two independent environments. Complete outputs are in [`results/`](results/).

**Detector** (development set)

| Instance | Sentence F1 | Sentence accuracy | Span F1 |
|---|---|---|---|
| Pipeline detector | 0.689 | 0.886 | 0.349 |
| Independent evaluation detector | 0.682 | 0.891 | 0.333 |

**Full system**

| System | Frac. neutral ↑ | Bias-prob. drop ↑ | SBERT ↑ | BERTScore F1 ↑ | SARI ↑ | BLEU | P(grammatical) ↑ |
|---|---|---|---|---|---|---|---|
| Copy input (baseline) | 0.603 | 0.000 | 1.000 | 1.000 | 100.0 | 100.0 | 0.906 |
| Delete flagged word (baseline) | 0.667 | 0.032 | 0.990 | 0.993 | 85.8 | 97.4 | 0.860 |
| Strategy A — WNC transfer | 0.923 | 0.198 | 0.218 | 0.805 | 7.4 | 0.4 | 0.300 |
| Strategy A — in-domain adapted | 0.907 | 0.189 | 0.251 | 0.813 | 9.0 | 0.4 | 0.238 |
| Strategy C — LEWIS-lite | **0.974** | **0.214** | 0.186 | 0.812 | **12.1** | 0.0 | **0.390** |

**Key findings**

- **Substantial bias reduction.** Editing raises the proportion of sentences
  judged neutral from 60% to as much as 97%. The mean drop in bias
  probability is about 6–7 times that of the word-deletion baseline, and the
  improvement is statistically significant for every editor (McNemar, p < 10⁻⁶).
- **No parallel corpus required.** The fully unsupervised Strategy C
  outperforms the editor pretrained on 53,803 Wikipedia edit pairs on
  transfer strength, SARI and grammaticality.
- **Consistent gains from in-domain adaptation.** Continued training on news
  pairs improved SARI by 18% and 21% in two independent runs, with gains in
  BERTScore in both.
- **Efficient.** The complete pipeline trains on a single free-tier GPU.

## Design decisions

- **Baseline:** the WNC-pretrained Strategy A checkpoint serves as the
  Pryzant et al. baseline, since the original release depends on an obsolete
  software stack.
- **Copy bias:** decoding boosts tokens from the unflagged part of the source
  sentence, rather than using a trained pointer-generator.
- **Strategy C:** mask positions come from the Stage 1 detector, which also
  serves as the LEWIS tagger.
- **Scope:** only lexical bias is modelled. BASIL's informational-bias
  annotations are excluded.
- **Synthetic data (Strategy B):** generated pairs must lower the independent
  detector's bias score and meet a semantic-similarity threshold. They are
  used for training only.

## Limitations and future work

The reported experiments cover Strategies A, A-adapted and C, each with a
single seed. Strategy B and a QLoRA decoder arm are implemented but were not
included in the reported runs. Improving adherence to the source wording,
through stronger copy mechanisms or greedy decoding, is the main direction
for future work.

## References

- Pryzant et al. (2020). *Automatically Neutralizing Subjective Bias in Text.* AAAI.
- Spinde et al. (2021). *Neural Media Bias Detection Using Distant Supervision With BABE.* Findings of EMNLP.
- Fan et al. (2019). *In Plain Sight: Media Bias Through the Lens of Factual Reporting.* EMNLP.
- Recasens et al. (2013). *Linguistic Models for Analyzing and Detecting Biased Language.* ACL.
- Reid & Zhong (2021). *LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer.* Findings of ACL.

BABE and BASIL are used under their research licenses. The lexicons in
`data/lexicons/` are redistributed under the MIT license (`data/lexicons/LICENSE`).
