# Implementation Flow

**Project:** Transformer-Based Sentiment and Emotion Analysis for Social Media Brand Intelligence
**Author:** Shahzaib Rashid

This document explains how the implementation is organised, what each step does,
and how to reproduce every result in the report from scratch.

---

## 1. Project structure

```
code/
├── data/
│   ├── raw/                  <- original datasets (downloaded, untouched)
│   └── processed/            <- cleaned copies used for training
├── notebook/                 <- the pipeline, one notebook per stage (run in order)
│   ├── download_data.ipynb
│   ├── data_preprocessing.ipynb
│   ├── data_eda.ipynb
│   ├── models.ipynb
│   └── evaluation.ipynb
├── models/                   <- fine-tuned models (created by models.ipynb)
│   └── deberta-base-local/   <- pre-trained DeBERTa weights (local copy)
├── results/                  <- every table and figure used in the report
│   ├── eda/                  <- data-analysis charts
│   ├── experiments.csv       <- master results table (one row per run)
│   ├── summary_table.csv     <- compact model x task F1 table
│   ├── efficiency.csv        <- parameters / latency / throughput
│   ├── figures_catalog.csv   <- list of all figures with descriptions
│   └── *.png                 <- comparison figure, learning curves, confusion matrices, ...
└── IMPLEMENTATION_FLOW.md    <- this file
```

---

## 2. The pipeline (run the notebooks in this order)

```
download_data  ->  data_preprocessing  ->  data_eda
                                             |
                                          models  ->  evaluation
```

### Step 1 - `download_data.ipynb`  (~1 minute)
Downloads the two benchmark datasets from the Hugging Face Hub and saves them
into `data/raw/` with their **official train/validation/test splits**:

| Local name | Source | Task |
|------------|--------|------|
| `tweeteval_sentiment` | cardiffnlp/tweet_eval (sentiment) | 3-class sentiment, Twitter |
| `goemotions` | google-research-datasets/go_emotions | 28-label emotion (multi-label), Reddit |

Keeping the official splits means no test data can influence any modelling
decision, and results stay comparable with published work.

### Step 2 - `data_preprocessing.ipynb`  (~1 minute)
Cleans the text and saves the result to `data/processed/`. Cleaning is minimal
on purpose (transformers were pre-trained on raw text):

1. links -> `http`
2. @usernames -> `@user` (privacy, and matches TweetEval's own convention)
3. character floods squeezed: `soooo` -> `soo`
4. whitespace normalised
5. **emojis and hashtags are kept** - they carry the emotional signal

### Step 3 - `data_eda.ipynb`  (~1 minute)
Explores the data and saves charts to `results/eda/`:

- class distribution (sentiment) and label frequency (emotion)
  -> shows the strong class imbalance -> justifies **macro-F1** as the main metric
- labels-per-comment chart -> proves GoEmotions is genuinely **multi-label**
  -> justifies sigmoid + binary cross-entropy instead of softmax
- text-length histograms -> 95% of posts are under ~27 words
  -> justifies the **64-token** input limit

### Step 4 - `models.ipynb`  (several hours - the training stage)
Fine-tunes three transformer encoders on both tasks (6 runs):

| Model | Checkpoint | Role in the comparison |
|-------|-----------|------------------------|
| DistilBERT | distilbert-base-uncased | efficiency end of the spectrum |
| RoBERTa | roberta-base | optimised pre-training, accuracy |
| DeBERTa | microsoft/deberta-base | disentangled attention, accuracy |

Training setup (identical for every model, matches report Section III.D):

- optimiser **AdamW**, learning rate **2e-5**, linear schedule with **6% warm-up**
- **same effective batch size 64** for every model
  (bigger models use gradient accumulation: 64x1, 32x2, 16x4)
- **early stopping** with patience 2, best epoch selected on the **validation** set
- max 4 epochs (sentiment) / 6 epochs (emotion), fixed **seed 42**
- single-label head: softmax + cross-entropy
- multi-label head: **sigmoid + binary cross-entropy**, labels as multi-hot vectors
- for multi-label, the decision **threshold is tuned on the validation set**
  (0.05-0.50 sweep) and then applied once to the test set
- GPU memory is freed after every run (`torch.mps.empty_cache()`) so the runs
  don't crash each other on 16 GB machines

Each run saves: the fine-tuned model (`models/<model>_<task>/`), a learning
curve, a confusion matrix (single-label), and appends one row to
`results/experiments.csv`.

**Practical notes**
- `QUICK_TEST = True` at the top runs the whole notebook on small samples
  in ~15 minutes to verify everything works. Set it to `False` for real results.
- The run-all cell **skips runs that are already in the results file**, so an
  interrupted session can simply be re-run and it continues where it stopped.
- On a laptop: keep it plugged in with the lid open.

### Step 5 - `evaluation.ipynb`  (~15 minutes)
Produces every analysis in the report's results section:

1. **Classical baseline** - TF-IDF + Linear SVM on both tasks (context for the
   transformer improvement: +13 to +16 macro-F1 points)
2. **Comparison figure + summary table** - `final_comparison.png`, `summary_table.csv`
3. **Efficiency** - parameters, single-text latency, batch throughput per model
4. **Robustness** - each architecture's F1 across the two datasets and the
   relative drop from the easy task to the hard one (`robustness_across_datasets.png`)
5. **RQ2 analysis** - runs the sentiment AND emotion models on the same posts and
   shows the distinct emotions hiding inside each sentiment class
   (`rq2_emotions_within_sentiment.png`) - the evidence that emotion detection
   adds business value beyond polarity

---

## 3. Final results (full data, seed 42)

| Model | Sentiment macro-F1 | Emotion macro-F1 | Latency | Throughput |
|-------|--------------------|------------------|---------|-----------|
| SVM baseline | 0.561 | 0.345 | - | - |
| DistilBERT | 0.688 | 0.438 | 5.2 ms | 1747 texts/s |
| **RoBERTa** | **0.720** | 0.458 | 9.1 ms | 1162 texts/s |
| **DeBERTa** | 0.701 | **0.469** | 11.8 ms | 1179 texts/s |

Key findings:

- RoBERTa is the best sentiment model (matches the published TweetEval level)
- DeBERTa is the best emotion model (above the published GoEmotions baseline)
  and the most robust across the two datasets
- DistilBERT keeps ~96% of RoBERTa's sentiment quality at ~1.7x the speed -
  the practical choice for high-volume monitoring

---

## 4. Environment

- Python 3.12 (Anaconda) with: `torch`, `transformers`, `datasets`,
  `scikit-learn`, `pandas`, `matplotlib`
- Hardware used: Apple M2 Pro, 16 GB RAM (GPU via the MPS backend);
  any CUDA GPU works the same way
- To run a notebook from the command line:
  `jupyter nbconvert --to notebook --execute --inplace notebook/<name>.ipynb`

## 5. Reproducing everything from zero

```
1. open notebook/download_data.ipynb      -> Run All   (~1 min)
2. open notebook/data_preprocessing.ipynb -> Run All   (~1 min)
3. open notebook/data_eda.ipynb           -> Run All   (~1 min)
4. open notebook/models.ipynb             -> set QUICK_TEST=False -> Run All (hours)
5. open notebook/evaluation.ipynb         -> Run All   (~15 min)
```

All tables and figures for the report are then in `results/`
(see `results/figures_catalog.csv` for a description of every image).
