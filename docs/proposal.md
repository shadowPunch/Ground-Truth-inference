# Efficient Detection and Neutralization of Lexical Political Bias in News Text under Compute Constraints

**A research proposal**

---

## Executive summary

This project builds a **detect-then-edit** system that identifies and neutralizes *lexical* political bias in news at the sentence level, trainable and deployable entirely on free-tier compute (Kaggle P100 / dual-T4). It adapts the methodology of Pryzant et al. (2020, *Automatically Neutralizing Subjective Bias in Text*) — specifically its **modular localize-then-edit design** and its edit-based inductive bias — while deliberately *dropping* the parts of that work that do not transfer to news: the concurrent end-to-end seq2seq model and the reliance on the Wikipedia Neutrality Corpus (WNC) as primary training data.

The central technical obstacle is that **no parallel biased→neutralized corpus exists for news**. Every available news-bias dataset (BABE, BASIL, MBIC, Ground News) supplies *labels and spans*, not rewrite targets. This proposal therefore treats **"where the neutral target comes from" as the primary experimental variable**, evaluating three strategies (WNC transfer, LLM-synthesized pseudo-parallel data, fully unsupervised editing) rather than assuming a single data source. Evaluation uses the standard style-transfer triad — **transfer strength × content preservation × fluency** — reported jointly rather than as a single scalar, with strong trivial baselines to keep the numbers honest.

---

## 1. Problem statement and motivation

Media coverage shapes public perception, and outlets encode bias partly through **word choice** — loaded verbs, epithets, and framing terms ("regime" vs. "government", "slammed" vs. "criticized", "terrorist" vs. "militant"). Automatically flagging and neutralizing this lexical layer supports media literacy, journalistic self-editing, and downstream NLP fairness.

Two framing commitments distinguish this project:

1. **Accessibility as a first-class constraint.** The system must train and run on free GPUs (Kaggle), not high-end hardware. This is not a limitation to apologize for — it is a design target that forces parameter-efficient methods and disciplined engineering, and makes the resulting system reproducible by anyone.
2. **Honesty about scope.** "Neutralizing ideological bias in news" in the general sense is close to ill-posed (see §2). We attack the well-posed sub-problem — sentence-level lexical bias — and are explicit about what we do *not* claim to solve.

---

## 2. Scope and task definition

### 2.1 The lexical / informational distinction

The BASIL dataset (Fan et al., 2019) draws the load-bearing distinction:

- **Lexical bias** — bias introduced through *word choice* within a sentence. This is local, token-level, and editable in place. **This is our target.**
- **Informational bias** — bias introduced through *which facts, quotes, and events are selected and how they are ordered*. This is a document- and corpus-level phenomenon. **You cannot fix it with a token edit**, because the biased unit is the *presence and selection* of content, not a swappable word.

Pryzant's method — a copy-heavy editor that changes a small span while leaving the rest of the sentence intact — is the *right* inductive bias for lexical bias and the *wrong* one for informational bias. We scope accordingly.

### 2.2 In scope / out of scope

| | In scope | Out of scope |
|---|---|---|
| **Bias type** | Lexical (word-choice) political bias | Informational / selection bias |
| **Granularity** | Single sentence | Full-article rewriting |
| **Operation** | Localize biased span → replace with neutral phrasing | Fact insertion/removal, reordering, summarization |
| **Claim** | "Reduces loaded-word framing while preserving meaning and fluency" | "Removes all ideological slant from an article" |

Informational bias is acknowledged as the complementary half of the problem and is addressed only by *positioning* — we reference NeuS-style neutral multi-document summarization (Lee et al., 2022) as the appropriate tool for that half and treat it as related work / future extension, **not** a second system to be built here.

### 2.3 Task formalization

Given a news sentence *x*:

1. **Detection / localization**: predict a binary sentence label (biased / neutral) and, if biased, tag the biased token span(s) *s ⊆ x*.
2. **Neutralization / editing**: produce *x′* that removes the lexical bias in *s* while preserving the propositional content and fluency of *x*.

This is exactly Pryzant's **modular** variant (a BERT-based detector feeding an editor), which we retain, as opposed to the **concurrent** joint model, which we discard because it requires parallel news pairs that do not exist.

---

## 3. Related work and positioning

### 3.1 What we keep and drop from Pryzant et al. (2020)

- **Keep**: the modular detect-then-edit architecture; the copy-heavy, local-edit inductive bias; the trivial baselines (copy-input, delete-flagged-word); WNC's single-word subset (55k pairs; the full corpus is 180k) as an *optional pretraining* resource.
- **Drop**: the concurrent end-to-end model (needs news pairs); WNC as *primary* corpus (it is Wikipedia — the domain shift we are trying to escape); any pretense that off-the-shelf bias datasets supply rewrite targets.

**Novelty delta vs. Pryzant** = precisely the replaced parts: (a) a news-domain bias-span detector trained on BABE/BASIL, (b) a principled, ablated strategy for obtaining neutral supervision in a domain with no parallel corpus, and (c) a three-way evaluation adapted to political neutralization.

### 3.2 Detection datasets and models (feed Stage 1)

- **BABE** (Spinde et al., 2021) — 3,700 expert-annotated sentences, word- and sentence-level bias labels; the standard sentence-level benchmark. A 2025 RoBERTa fine-tune on BABE is a strong, reproducible baseline target.
- **BASIL** (Fan et al., 2019) — 300 articles / ~7,984 sentences with span-level annotation separating informational from lexical bias; supplies the token-level supervision for span localization.
- **MBIC** (Spinde et al., 2021) — 1,700 crowd-annotated sentences with annotator metadata (BABE's precursor).
- **MBIB** (Wessel et al., 2023) — a unified benchmark of 9 bias tasks / 22 datasets; useful for standardized evaluation. Note its finding: **political bias is substantially harder to detect than hate speech or gender bias** — a realistic expectation-setter.
- **POLITICS** (Liu et al., 2022) — a RoBERTa-scale encoder pretrained by contrasting same-story articles across the ideological spectrum (BIGNEWS, 3.6M articles). Excellent warm-start encoder for the detector; note the authors' caveat that it must be fine-tuned on a downstream task before use.

### 3.3 Neutralization / editing precedents

- **NeuS** (Lee et al., 2022) — neutral multi-document summarization over left/center/right articles, supervised by AllSides expert roundups. The right tool for *informational* bias; ~3,564 triplets, but link rot leaves roughly ~1,766 usable after re-scraping — a practical warning.
- **Sentiment-polarity neutralization in news** (arXiv 2402.02145, 2024) — sentence-level, explicitly notes Pryzant's single-word limitation and the difficulty of assembling a parallel news corpus, and therefore uses LLM-based contextual perturbation instead. Direct precedent for our Strategy B.
- **"Neutralizing the Narrative"** (arXiv 2504.03520, 2025) — two-stage LLM detect-then-debias over 30k crime articles from politically diverse sources.
- **BIASsist** (CHI 2025) — reader-facing bias identification, explanation, and neutralization.

### 3.4 Non-parallel style transfer (the methodological backbone for Strategies B/C)

These provide the machinery for editing *without* a parallel corpus, and each maps naturally onto detect-then-edit:

- **Delete-Retrieve-Generate** (Li et al., 2018) — delete style-marked phrases, retrieve target-style phrases, generate.
- **Mask-and-Infill** (Wu et al., 2019) — mask style-associated tokens, infill with a pretrained MLM. Maps *directly* onto "detector localizes span → MLM infills neutral replacement."
- **Masker** (Malmi et al., 2020) — train MLMs on source (biased) and target (neutral) domains, delete where they disagree most in likelihood, infill with a padded MLM; fully unsupervised, no parallel data.
- **LEWIS** (Reid et al., 2021) — RoBERTa tagger (insert/replace/delete) + BART generator; uses classifier-driven masking and style-specific LMs to *synthesize pseudo-parallel data*. The natural modern successor to Pryzant for the non-parallel setting.

---

## 4. Data strategy (the core methodological risk)

### 4.1 What each source actually provides

| Source | Grain | Provides | Role here | Access note |
|---|---|---|---|---|
| **WNC** (Pryzant) | Sentence pairs | biased→neutral parallel (Wikipedia) | Editor **pretraining** only | `rpryzant/neutralizing-bias` |
| **BABE** | Sentence + word | biased/neutral label, biased words | Detector training / bias classifier | Media Bias Group / HF |
| **BASIL** | Sentence + span | lexical vs informational spans | Detector span supervision | Fan et al. repo |
| **MBIC** | Sentence + word | crowd bias labels | Auxiliary detector data | Media Bias Group |
| **MBIB** | Mixed | 9-task unified benchmark | Standardized eval | `mediabiasgroup/mbib` |
| **NeuS / AllSides** | Story triplets | L/C/R articles + neutral roundup | Informational-bias reference / future work | Lee et al. repo (link rot) |
| **POLITICS / BIGNEWS** | Article | ideology-pretrained encoder + corpus | Detector backbone; neutral-side LM data | `launchnlp/POLITICS` |
| **Ground News** | Source / story | source-level lean, same-story clusters | Distant supervision / clustering **only** | Consumer product — **verify API/licensing** |
| **NELA-GT** | Article | source-level reliability/lean labels | Distant supervision at scale | Public research release |

**The gap, restated plainly:** none of these gives a *news* biased-sentence paired with its *human-neutralized* rewrite. That target must be manufactured, and how we manufacture it is the experiment.

### 4.2 Three target-data strategies (evaluated as ablation arms)

**Strategy A — WNC transfer + domain adaptation.**
Pretrain a small seq2seq editor on WNC's single-word subset, then adapt to news (continued training on any pseudo-pairs from B, or unsupervised objectives from C). *Pro*: real human edits. *Con*: Wikipedia→news domain shift; encyclopedic register ≠ journalistic register.

**Strategy B — LLM-synthesized pseudo-parallel corpus.**
Take biased news sentences (BABE/BASIL positives), and have one or more strong LLMs (e.g., Gemini, Claude) generate neutralized targets *offline* (no GPU cost). Train the small editor on the resulting pseudo-parallel pairs. *Pro*: in-domain, scalable. *Con*: **circularity** — "neutral" becomes "whatever the LLM thinks is neutral," and LLM "neutral" text carries measurable partisan skew (cf. arXiv 2410.09978). Mitigations are mandatory (§4.3).

**Strategy C — Fully unsupervised editing.**
No targets at all. Use the detector to mask biased spans and infill with a neutral-domain MLM (Mask-and-Infill / Masker), or train a LEWIS-style tagger+generator using center-outlet news as the neutral corpus and biased-outlet news as the source. *Pro*: no synthetic-target circularity, no parallel data needed. *Con*: lower controllability; harder to hit fluency.

These are not mutually exclusive; A can initialize C, and B can supply pseudo-pairs to stabilize either. But each is run and reported **as its own arm** so the effect of the target-source choice is measurable rather than hidden.

### 4.3 Data governance and integrity

- **Synthetic data never touches evaluation.** All LLM-generated pairs are training-side only. The test set is drawn from human-labeled data (BABE/BASIL held-out) with a bias classifier that is *independent* of any generation model.
- **"Neutral" is contested for political content.** Unlike Wikipedia NPOV, there is no editorial consensus target for news. We document our operational definition of neutral, use *multiple* independent bias signals where possible, and report sensitivity to that definition rather than treating it as ground truth.
- **Circularity mitigations for Strategy B**: generate with ≥2 distinct model families; filter generated targets through the *independent* BABE-trained classifier (reject targets that don't actually reduce bias); reject targets whose semantic similarity to the source falls below a threshold (guards against content drift); human spot-check a random sample and report agreement.
- **Licensing**: confirm redistribution terms for each dataset; Ground News and any scraped news text require an explicit licensing/ToS check before use — prefer released research corpora (NELA-GT, BIGNEWS) over scraping.

---

## 5. Methodology and system architecture

```
          ┌────────────────────────────────────────────────────┐
  news    │  STAGE 1 — DETECTOR (localization)                  │
 sentence │  POLITICS / RoBERTa-base encoder                    │
   x  ───▶│  • sentence head: biased vs neutral                 │
          │  • token head: BIO tag of biased span(s) s          │
          │  trained on BABE (sentence) + BASIL (span)          │
          └───────────────┬────────────────────────────────────┘
                          │  s (biased span)
                          ▼
          ┌────────────────────────────────────────────────────┐
  x′  ◀───│  STAGE 2 — EDITOR (neutralization)                  │
 neutral  │  small seq2seq (T5-small/base, BART-base) OR        │
 sentence │  MLM infiller OR QLoRA'd small decoder              │
          │  supervision = Strategy A / B / C                   │
          └────────────────────────────────────────────────────┘
```

### 5.1 Stage 1 — Detector

- **Backbone**: POLITICS (ideology-aware, RoBERTa-large scale) as primary; RoBERTa-base / DistilRoBERTa as lighter fallbacks.
- **Heads**: a sentence-level classification head (BABE) and a token-classification (BIO) head for span localization (BASIL word-level spans; BABE biased-word annotations).
- **Why full fine-tune is fine here**: at 125M–355M params, an encoder fine-tunes comfortably within a single P100/T4 with mixed precision + gradient checkpointing; PEFT is optional, not required, at Stage 1.
- **Doubles as evaluator**: a *separately trained* instance of this classifier provides the Aggregate Bias Score at evaluation time (train/eval instances must not share data).

### 5.2 Stage 2 — Editor

- **Primary editor**: T5-small (60M) or BART-base (140M) seq2seq — small enough to *fully* fine-tune on free GPUs, copy-friendly for local edits.
- **Unsupervised infiller (Strategy C)**: RoBERTa-based MLM for Mask-and-Infill / Masker; optionally a LEWIS-style RoBERTa-tagger + BART-generator.
- **Stretch editor (ambitious arm)**: a 1–3B instruction-tuned decoder (e.g., Qwen2.5-1.5B, Gemma-2-2B) fine-tuned with **QLoRA** (4-bit NF4) for higher fluency; feasible on a single T4 (§7).
- **Constrained decoding**: bias the decoder toward copying non-flagged tokens (span-restricted editing) so meaning is preserved and the edit stays local — the operational core of Pryzant's inductive bias.

---

## 6. Evaluation protocol

Style transfer is only meaningfully evaluated on **three axes jointly**. Any single axis is trivially gameable.

### 6.1 The triad

1. **Transfer strength (bias reduction)** — *Aggregate Bias Score*: fraction of outputs flagged neutral by an **independent** BABE-trained classifier, plus the mean drop in its bias probability (source → output). Independence from the training signal is essential; watch for Goodhart once optimizing against it.
2. **Content preservation** — *Semantic Similarity*: BERTScore and SBERT cosine between *x* and *x′*; plus SARI and BLEU (SARI is more appropriate than BLEU for edit tasks because it rewards correct keeps/deletes/adds). Preservation alone rewards the copy-input degenerate solution, so it is *never* read in isolation.
3. **Fluency** — perplexity under a **fixed** external LM (e.g., GPT-2) and/or a CoLA-based grammaticality classifier. This is the guardrail that stops the model from "neutralizing" by mangling the sentence.

### 6.2 Joint reporting

Report the three axes together, and plot **bias-reduction vs. preservation as a Pareto frontier** across decoding settings, rather than collapsing to one number. The whole point of the task is the trade-off; a single aggregate hides it.

### 6.3 Baselines (mandatory honesty checks)

- **Copy-input** (change nothing) — upper bound on preservation, zero transfer.
- **Delete-flagged-word** — Pryzant's trivial editor; annoyingly competitive on this task.
- **Pryzant off-the-shelf (WNC-trained)** — measures how far pure Wikipedia transfer gets on news.
- **LLM zero-shot neutralization** — a strong-but-not-free reference ceiling; contextualizes what a small, free-GPU system gives up.

### 6.4 Human evaluation

A small (~150–300 item) human study rating bias-removal, meaning-preservation, and fluency on Likert scales, with inter-annotator agreement reported. Kept small deliberately so it is feasible solo.

### 6.5 Statistical rigor

Given the emphasis on avoiding the methodological flaws that sink bias-NLP papers: fixed and documented train/dev/test splits with **no story or outlet leakage across splits** (same-story sentences must not straddle the split boundary); ≥3 seeds with mean ± std; paired significance testing (McNemar for the classifier, bootstrap CIs for generation metrics) against baselines. Report negative results (e.g., if a strategy fails to beat delete-flagged-word).

---

## 7. Compute plan — training under free-tier constraints

### 7.1 The hardware envelope (Kaggle, as of 2026)

| Resource | Spec |
|---|---|
| GPU option A | 1× **P100** 16 GB HBM2, ~732 GB/s, Pascal — **no tensor cores** (weaker mixed-precision) |
| GPU option B | **2× T4** 16 GB GDDR6 each, Turing — tensor cores + INT8 |
| Weekly quota | ~30 GPU-hours |
| Session cap | up to ~9–12 h per run |
| Persistent storage | ~20 GB (`/kaggle/working`) |
| CPU RAM | ~32 GB |

Implication: **P100 for compute-bound single-GPU training; T4×2 when tensor-core mixed precision or two-GPU data parallel helps.** Everything must survive a session kill, because the weekly quota and 12-h cap guarantee multi-session training.

### 7.2 Memory-reduction stack

Applied roughly in order of impact:

1. **Parameter-efficient fine-tuning (LoRA/QLoRA)** — for the stretch decoder editor. **QLoRA** = 4-bit **NF4** quantization of the frozen base + **double quantization** (compresses quantization constants, ~0.5 bit/param saved) + **LoRA adapters** trained in bf16 + **paged optimizers** (spill optimizer state to CPU RAM to survive spikes). A 7B base sits in ~4–5 GB of 4-bit weights; a 1–3B decoder is very comfortable on one T4. Encoders and small seq2seq don't need this — full fine-tune fits.
2. **Gradient checkpointing** — recompute activations in the backward pass instead of storing them. Large activation-memory savings for a modest (~20–30%) compute cost; the single most important lever after quantization for fitting longer sequences.
3. **Mixed precision** — bf16 preferred on T4 (tensor cores); on P100 (no tensor cores) fp16 gives memory savings but limited speedup, so P100 runs are chosen for memory headroom, not throughput.
4. **8-bit optimizer** (bitsandbytes `adamw_8bit`) — Adam's two moment buffers can exceed the weights in memory; 8-bit states cut that sharply.
5. **Gradient accumulation** — simulate large effective batch sizes with batch size 1–4 to stay within VRAM.
6. **Sequence-length capping + dynamic padding + length bucketing** — sentence-level task means short sequences (cap ~128 tokens); pad to batch-max, not corpus-max; bucket by length to minimize wasted compute.
7. **Data streaming + on-disk datasets** — stream/tokenize on the fly with cursor-based resumption to avoid CPU-RAM blowups on large corpora.

### 7.3 Model-size budget

| Component | Model | Params | Fits free GPU? | Method |
|---|---|---|---|---|
| Detector | POLITICS / RoBERTa-base | 355M / 125M | Yes, easily | Full fine-tune + bf16 + checkpointing |
| Editor (primary) | T5-small / BART-base | 60M / 140M | Yes, easily | Full fine-tune |
| MLM infiller (C) | RoBERTa-base | 125M | Yes | Full fine-tune / off-the-shelf |
| Editor (stretch) | Qwen2.5-1.5B / Gemma-2-2B | 1.5–2B | Yes on T4 | **QLoRA** 4-bit + paged AdamW |

**None of the core pipeline requires QLoRA** — that's the point. QLoRA is reserved for the optional high-fluency decoder arm. The core detector + small-seq2seq editor is well within full fine-tuning on a single free GPU.

### 7.4 Checkpoint-and-resume discipline

Because training *will* span multiple sessions: checkpoint model + optimizer + scheduler + data-cursor state to `/kaggle/working` (or a Kaggle Dataset for >20 GB or cross-notebook reuse) every N steps; on session start, detect and resume from the latest checkpoint; keep steps-per-session well under the cap with margin for the final checkpoint write. (This mirrors a streaming/cursor-resume pattern that is already a solved engineering problem.)

---

## 8. Work plan

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Setup & data audit | 1–2 | Datasets acquired, licenses confirmed, splits frozen (no story/outlet leakage), baselines (copy, delete-word) implemented |
| 1. Detector | 3–5 | POLITICS/RoBERTa detector: sentence + span heads; benchmarked on BABE/BASIL; **this instance also becomes the independent evaluator (separate split)** |
| 2. Editor — Strategy A | 6–7 | WNC-pretrained small seq2seq; measured off-the-shelf on news |
| 3. Editor — Strategy B | 8–10 | LLM-synth pseudo-parallel corpus (with §4.3 mitigations); in-domain editor |
| 4. Editor — Strategy C | 11–13 | Mask-and-Infill / Masker / LEWIS unsupervised editor |
| 5. Evaluation | 14–15 | Full triad + Pareto plots + baselines + significance tests + small human study |
| 6. Analysis & writeup | 16 | Ablation of target-data strategy; error analysis; limitations; report |

Single-person, free-GPU-realistic. Strategies B and C can be reordered or run in parallel across the weekly quota.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| No parallel news corpus (core risk) | Certain | Three-strategy ablation; never assume a single source |
| Strategy-B circularity (LLM defines "neutral") | High | Multi-model generation, independent-classifier filtering, semantic-similarity floor, human spot-check, synthetic never in eval |
| Model games one metric (mangles text) | Medium | Triad reported jointly; fluency guardrail; Pareto reporting |
| Trivial baseline (delete-word) matches model | Medium | Report it honestly; if unbeaten, that *is* a finding |
| Compute quota exhaustion mid-train | Medium | Checkpoint/resume; small models; PEFT only where needed |
| Split leakage inflates scores | Medium | Story/outlet-disjoint splits enforced at phase 0 |
| Political sensitivity / annotator subjectivity | Inherent | Document operational neutrality definition; report sensitivity; lean on expert-annotated BABE |
| Dataset link rot (NeuS/AllSides) | Known | Treat NeuS as related work, not a dependency |

---

## 10. Expected contributions and deliverables

1. **A reproducible, free-GPU-trainable detect-then-edit pipeline** for sentence-level lexical political-bias neutralization.
2. **An empirical answer to "where should neutral supervision come from?"** — a controlled comparison of WNC-transfer vs. LLM-synthesized vs. unsupervised target strategies, which the literature currently conflates or leaves implicit.
3. **A political-neutralization evaluation harness** (transfer × preservation × fluency, with strong trivial baselines and leakage-safe splits) that others can reuse.
4. **A clear, documented scope boundary** between lexical (editable) and informational (non-editable-by-token) bias, positioning NeuS-style summarization as the complementary tool.
5. Open code, model weights, and the (training-only) synthetic corpus, with full licensing provenance.

**Explicit non-goals:** removing informational/selection bias; full-article rewriting; claiming a single objective definition of political neutrality.

---

## References

- Pryzant et al. (2020). *Automatically Neutralizing Subjective Bias in Text.* arXiv:1911.09709 — WNC (180k pairs; 55k single-word subset); modular & concurrent models.
- Fan et al. (2019). *In Plain Sight: Media Bias Through the Lens of Factual Reporting* (BASIL). — informational vs. lexical bias, span-level.
- Spinde et al. (2021). *Neural Media Bias Detection Using Distant Supervision With BABE.* arXiv:2209.14557 — 3,700 expert-annotated sentences.
- Spinde et al. (2021). *MBIC — A Media Bias Annotation Dataset Including Annotator Characteristics.*
- Wessel et al. (2023). *MBIB — The Media Bias Identification Benchmark.* — 9 tasks / 22 datasets; political bias is the hard tier.
- Liu et al. (2022). *POLITICS: Pretraining with Same-story Article Comparison for Ideology Prediction and Stance Detection.* arXiv:2205.00619 — BIGNEWS, 3.6M articles.
- Lee et al. (2022). *NeuS: Neutral Multi-News Summarization for Mitigating Framing Bias.* arXiv:2204.04902 — AllSides roundups; ~3,564 triplets.
- Spinde et al. (2023). *The Media Bias Taxonomy: A Systematic Literature Review.* arXiv:2312.16148 — orientation survey.
- (2024). *Analyzing Sentiment Polarity Reduction in News... Contextual Perturbation and LLMs.* arXiv:2402.02145 — sentence-level news neutralization; Pryzant limitation.
- (2025). *Neutralizing the Narrative: AI-Powered Debiasing of Online News Articles.* arXiv:2504.03520 — two-stage detect-then-debias.
- (2025). *BIASsist: Empowering News Readers via Bias Identification, Explanation, and Neutralization.* CHI 2025.
- Li et al. (2018). *Delete, Retrieve, Generate.* NAACL — edit-based non-parallel transfer.
- Wu et al. (2019). *Mask and Infill.* IJCAI — MLM span infilling.
- Malmi et al. (2020). *Unsupervised Text Style Transfer with Padded Masked Language Models (Masker).* arXiv:2010.01054 — fully unsupervised delete+infill.
- Reid & Zhong (2021). *LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer.* — tagger + generator, pseudo-parallel synthesis.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314 — NF4, double quantization, paged optimizers.

*Note: dataset access paths and licenses should be re-verified at project start; Ground News in particular is a consumer product whose data access/terms must be confirmed before any use.*
