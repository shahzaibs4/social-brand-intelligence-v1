# Complete Project Guide
### Transformer-Based Sentiment and Emotion Analysis for Social Media Brand Intelligence
**Author: Shahzaib Rashid**

This one document explains the whole project from start to finish: why it exists,
which data it uses and why, how to download the data, how the experiments run,
how the models are fine-tuned, what the results are, and what was achieved.

---

## 1. What is the purpose of this project?

Social media produces millions of posts about brands every day. Companies cannot
read them by hand, so they use automatic tools. The problem is that most tools
only say whether a post is **positive, negative, or neutral**. That single label
is not enough for a business:

- an **angry** customer is about to leave and needs an urgent reply,
- a **disappointed** customer had expectations that were not met and can be won
  back with a follow-up,
- a **fearful** customer needs reassurance,
- a customer showing **disapproval** is pointing at a product or policy problem.

All four of these look identical to a normal sentiment tool: just "negative".

**The purpose of this project** is to build and evaluate a system that detects
**both** signals at the same time - the sentiment (how positive or negative a
post is) **and** the fine-grained emotion behind it (which of 28 emotional
states the writer expresses) - and to find out which modern transformer model
does this best, how fast each model is, and how all of it can be used by a real
company through a web dashboard.

The project answers three research questions:

1. **RQ1** - Which transformer model performs best for sentiment and for
   fine-grained emotion detection?
2. **RQ2** - Does adding emotion detection give useful information beyond
   sentiment alone?
3. **RQ3** - How consistently do the models perform across different platforms
   and task difficulties?

## 2. Which datasets are used, and why these ones?

| Dataset | Platform | Task | Size (train/val/test) |
|---------|----------|------|----------------------|
| **TweetEval (sentiment)** | Twitter | 3-class sentiment (negative / neutral / positive) | 45,615 / 2,000 / 12,284 |
| **GoEmotions** | Reddit | 28 emotion labels, **multi-label** (one comment can carry several emotions) | 43,410 / 5,426 / 5,427 |

**Why these two datasets were chosen:**

1. **They are the standard public benchmarks** for exactly our two tasks.
   Published papers report results on them, so our numbers can be compared
   directly with the literature - that is what makes results credible.
2. **They come with official train/validation/test splits.** Keeping the
   official splits means no test data can leak into training, and reviewers
   can trust the evaluation.
3. **They cover two different platforms** (Twitter and Reddit) with different
   writing styles - this lets us test robustness across platforms (RQ3).
4. **They cover both task types**: single-label classification (sentiment) and
   the much harder multi-label classification (emotion), so the comparison of
   models is meaningful.
5. **They are free, public, and ethically safe** - user names and links are
   masked, and no private data is used.

## 3. How to download the data

Open `notebook/download_data.ipynb` and run all cells. It downloads both
datasets from the Hugging Face Hub and stores them in `data/raw/`:

- TweetEval sentiment: `cardiffnlp/tweet_eval` (configuration "sentiment")
- GoEmotions: `google-research-datasets/go_emotions` (configuration "simplified")

The notebook skips anything already downloaded, prints the split sizes, and
shows one example from each dataset so you can see what the data looks like.
Total download is only ~12 MB.

## 4. How the data is prepared

`notebook/data_preprocessing.ipynb` cleans the text and saves it to
`data/processed/`. Cleaning is **minimal on purpose** - transformers were
pre-trained on raw text, so heavy cleaning removes useful signal:

1. web links become `http`
2. usernames become `@user` (privacy + matches TweetEval's own format)
3. character floods are squeezed: "soooo" becomes "soo"
4. extra whitespace is removed
5. **emojis and hashtags are kept** - they carry emotion

`notebook/data_eda.ipynb` then explores the data and produces the charts in
`results/eda/`. Three findings from this analysis shaped the whole method:

- the emotion labels are **heavily imbalanced** (some labels are 100x rarer
  than others) → we must use macro-F1, which counts every class equally;
- many comments carry **more than one emotion** → the task must be modelled
  as multi-label (independent yes/no per emotion), not as a single choice;
- 95% of posts are under 27 words → 64 tokens of input is enough, which
  halves the computation.

## 5. How the models are fine-tuned

`notebook/models.ipynb` fine-tunes three pre-trained transformers on both
tasks (6 runs in total):

| Model | Size | Why it is included |
|-------|------|--------------------|
| DistilBERT | 67M parameters | the speed end: compressed model for cheap deployment |
| RoBERTa | 125M | the accuracy end: improved BERT pre-training |
| DeBERTa | 139M | the architecture end: disentangled attention, state of the art |

**Why exactly these three models?**

The three models were not picked randomly - each one represents a different
*strategy* for improving the original BERT, so together they answer the
question "which direction of improvement matters most for our tasks?":

- **DistilBERT** represents **compression** (knowledge distillation). It keeps
  ~97% of BERT's language understanding in a model that is 40% smaller and 60%
  faster. If a compressed model is almost as good, companies should not pay
  for a big one - this hypothesis can only be tested by including one.
- **RoBERTa** represents **better training of the same architecture**. It is
  identical to BERT inside, but pre-trained longer, on more data, with dynamic
  masking. It shows how much performance comes purely from better pre-training.
- **DeBERTa** represents **a better architecture**. Its disentangled attention
  stores *what* a word is and *where* it is separately, which should help with
  fine-grained distinctions. It shows how much performance comes from
  architectural innovation.

Together they span the practical trade-off space (fast-and-small vs
accurate-and-large) AND the scientific one (compression vs training vs
architecture). A comparison of three near-identical models would answer
neither question.

**Why not other candidates?**

- **Plain BERT** - excluded because every included model is a documented
  improvement over it; it would only reproduce known results and DistilBERT
  already anchors the comparison to the BERT family.
- **Large language models (GPT-style)** - recent evaluations (cited in the
  paper, ref. [14]) show that *fine-tuned* small encoders remain equal or
  better on domain-specific sentiment tasks while costing a fraction to run.
  A brand monitor scoring millions of posts per day cannot afford LLM
  inference costs, so encoders are the right tool class for this use case.
- **Classical models** are not excluded - a TF-IDF + SVM baseline is included
  precisely so the transformer advantage is measured, not assumed.
- **Model sizes** - all three are "base" size (67-139M) so that the comparison
  is apples-to-apples and everything trains on a single consumer machine,
  which also makes the study fully reproducible for others.

**What fine-tuning means here:** each model arrives already pre-trained on
billions of words. We add a new classification head (dropout + linear layer)
and update **all** weights on our task data. The head differs by task:

- **Sentiment:** softmax over 3 classes, cross-entropy loss (one correct class);
- **Emotion:** 28 independent sigmoid outputs, binary cross-entropy loss
  (any combination of emotions can be correct at once).

**The exact settings, and why:**

| Setting | Value | Why |
|---------|-------|-----|
| Optimiser | AdamW | standard for transformers |
| Learning rate | 2e-5, linear decay | the standard fine-tuning value; small enough not to destroy pre-trained knowledge |
| Warm-up | first 6% of steps | protects the weights while the new random head settles |
| Effective batch size | 64 for every model | fairness - identical training dynamics (larger models use gradient accumulation: 64x1, 32x2, 16x4) |
| Max length | 64 tokens | justified by the EDA (95% of posts under 27 words) |
| Max epochs | 4 sentiment / 6 emotion | the 28-label task needs more passes for rare classes |
| Early stopping | patience 2, best epoch restored | stops when validation stops improving; DeBERTa-emotion stopped at epoch 5 |
| Threshold (emotion) | tuned 0.05-0.50 on validation | a fixed 0.5 cut-off predicts almost nothing for rare emotions |
| Seed | 42 | reproducibility |

**How to run it:** open `notebook/models.ipynb`. Set `QUICK_TEST = True` first
and Run All - this verifies the whole pipeline on tiny samples in ~15 minutes.
Then set `QUICK_TEST = False` and Run All for the real experiments (several
hours - run overnight, laptop plugged in). Every finished run is saved
immediately, so an interrupted session continues where it stopped when you
run it again. Fine-tuned models are saved into `models/`.

## 6. How the evaluation works

`notebook/evaluation.ipynb` (about 15 minutes) produces everything in `results/`:

- **Metrics** on the untouched official test sets: Accuracy, Precision, Recall,
  and **macro-F1** (the headline metric - every class counts equally);
- a **classical baseline** (TF-IDF + linear SVM) trained on the same data, so
  the transformer improvement is measured, not assumed;
- **efficiency**: parameters, single-text latency, throughput per model;
- **robustness**: how much each model drops from the easy task to the hard one;
- **the RQ2 analysis**: both best models score the same 2,000 posts, and we
  look at which emotions hide inside each sentiment class.

## 7. What was achieved - the results

**Main results (macro-F1 on official test sets):**

| Model | Sentiment | Emotion |
|-------|-----------|---------|
| TF-IDF + SVM (baseline) | 0.561 | 0.345 |
| DistilBERT | 0.688 | 0.438 |
| **RoBERTa** | **0.720** | 0.458 |
| **DeBERTa** | 0.701 | **0.469** |

**What these numbers mean - the five achievements:**

1. **Transformers clearly beat classical machine learning**: +13 to +16
   macro-F1 points over the SVM on both tasks, measured in our own controlled
   experiments.
2. **RoBERTa is the best sentiment model** (0.720) - matching the level
   reported in the published TweetEval literature, which confirms our whole
   pipeline is correct.
3. **DeBERTa is the best emotion model** (0.469) - *above* the published
   GoEmotions BERT baseline (~0.46), and it is also the most robust model:
   it loses the least performance (33%) when moving from the easy 3-class
   task to the hard 28-label task.
4. **Speed trade-off measured**: DistilBERT keeps 96% of the best sentiment
   quality while answering in 5.2 ms and processing ~1,750 texts per second -
   up to 2.3x faster than the big models. That is the model to choose for
   high-volume monitoring.
5. **Emotion genuinely adds value beyond sentiment (RQ2)**: on 2,000 unseen
   posts, the 772 posts labelled "negative" contained at least five clearly
   different emotions (disapproval 10%, annoyance 9%, sadness 6%, anger 5%,
   plus disappointment) - each needing a different business response. A
   sentiment-only tool merges all of them into one alert.

## 8. The web application

`app.py` turns the research into a working tool (`streamlit run app.py`):

- **About the Project** - what the project is and the key findings
- **Results** - every table and chart, each with a plain-English explanation
- **Fine-Tuning Details** - how the models were trained and why
- **Live Analysis** - type any post; RoBERTa gives the sentiment, DeBERTa the
  emotions, side by side with probability charts

## 9. Where everything is published

| Artefact | Location |
|----------|----------|
| Code (this repository) | github.com/shahzaibs4/social-brand-intelligence-v1 |
| Fine-tuned sentiment model | huggingface.co/code-world/roberta-sentiment-brand |
| Fine-tuned emotion model | huggingface.co/code-world/deberta-emotion-brand |
| All figures with descriptions | `results/figures_catalog.csv` |
| Full methodology details | `IMPLEMENTATION_FLOW.md`, `FINE_TUNE.md` |
| How to run on Windows | `README.md` |
| How to run only the web app | `RUN_WEBAPP.md` |

## 10. Reproducing the whole project from zero

```
1. notebook/download_data.ipynb       -> Run All   (~1 min)
2. notebook/data_preprocessing.ipynb  -> Run All   (~1 min)
3. notebook/data_eda.ipynb            -> Run All   (~1 min)
4. notebook/models.ipynb              -> QUICK_TEST=False, Run All  (hours)
5. notebook/evaluation.ipynb          -> Run All   (~15 min)
6. streamlit run app.py               -> the dashboard
```

## 11. Answers to the research questions

**RQ1 - Which transformer model performs best?**
There is no single winner, and that is itself the finding. **RoBERTa** is the
best sentiment model (macro-F1 0.720). **DeBERTa** is the best emotion model
(macro-F1 0.469). On the easy 3-class task the three models finish within 3.2
points of each other; on the hard 28-label task the architectural differences
become visible and DeBERTa's disentangled attention pays off. Practical answer:
RoBERTa for polarity, DeBERTa for emotions, DistilBERT when speed matters more
than the last few points.

**RQ2 - Does emotion detection add value beyond sentiment?**
Yes, measurably. When both best models scored the same 2,000 unseen posts, the
772 posts labelled "negative" turned out to contain at least five clearly
different emotional states (disapproval 10%, annoyance 9%, sadness 6%, anger
5%, plus disappointment). Each of these requires a different business action -
escalation, follow-up, product feedback - but a sentiment-only system collapses
all of them into one undifferentiated alert. This is the empirical core of the
project's argument.

**RQ3 - How consistent are the models across platforms and task difficulty?**
Every model loses roughly a third of its macro-F1 when moving from Twitter
3-class sentiment to Reddit 28-label emotion, but not equally: DistilBERT
drops 36.3%, RoBERTa 36.4%, and DeBERTa only 33.1%. DeBERTa is therefore the
most robust architecture - it never wins the easy task but degrades most
gracefully as difficulty rises, which matters for teams that must deploy one
model across varied workloads.

## 12. Honest limitations

1. **Single random seed (42).** The sentiment ranking margin is comfortable
   (1.9 points) but the emotion margin (1.1 points) could be sensitive to seed
   variance. The pipeline supports re-running with other seeds unchanged.
2. **One global decision threshold** for all 28 emotions. Per-class thresholds
   would likely raise the absolute emotion scores further.
3. **English only**, two platforms only. Nothing is known yet about other
   languages or platforms like Instagram comments or product reviews.
4. **DeBERTa v1, not v3** - the newer v3 was numerically unstable on the Apple
   Metal (MPS) training backend used here; it should be revisited on CUDA
   hardware.

## 13. How can this work be extended?

**Scientific extensions (strengthen the evidence):**

1. **Multi-seed replication** - repeat all runs with seeds 43 and 44 and report
   mean and standard deviation with a significance test. The training notebook
   needs only the seed value changed; everything else is identical.
2. **Per-class threshold optimisation** - tune one threshold per emotion on the
   validation set instead of a single global cut-off; the rare emotions gain most.
3. **Class-weighted or focal loss** - give rare emotions more weight during
   training to fight the long-tailed label distribution at the source.
4. **DeBERTa-v3 and newer encoders** (e.g. ModernBERT) on CUDA hardware - test
   whether newer architectures push the emotion ceiling further.
5. **LLM comparison** - benchmark GPT-class models in zero-shot and few-shot
   mode on the same test sets to quantify the cost/quality trade-off against
   fine-tuned encoders.

**Data extensions (widen the coverage):**

6. **Multilingual models** (XLM-RoBERTa, mDeBERTa) on non-English social media -
   brand perception is global.
7. **More platforms and domains** - Instagram, YouTube comments, product
   reviews, support tickets; cross-domain evaluation would show how far the
   models travel.
8. **Aspect-based analysis** - link each detected emotion to the specific
   product or service attribute that caused it ("angry *about delivery*",
   "disappointed *by battery life*") - the most valuable next step for business
   users.

**Engineering extensions (make it production-grade):**

9. **Real-time streaming** - connect the dashboard to live platform APIs and
   score posts as they arrive, with alerting when negative emotion spikes.
10. **Model compression of the winners** - distil or quantise the fine-tuned
    RoBERTa/DeBERTa to serve their accuracy at DistilBERT speed.
11. **A REST API service** around the two models so other company systems
    (CRM, ticketing) can request predictions programmatically.
12. **Human feedback loop** - let analysts correct wrong predictions in the
    dashboard and periodically fine-tune on the corrections.
