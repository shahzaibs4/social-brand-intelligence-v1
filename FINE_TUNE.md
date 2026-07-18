# Fine-Tuning Documentation

How the three transformer models (DistilBERT, RoBERTa, DeBERTa) were fine-tuned,
which settings and techniques were used, and **why** each choice was made.
The implementation is in `notebook/models.ipynb`.

---

## 1. How the datasets were split

We did **not** create our own splits - both datasets come with official
train / validation / test splits, and we kept them exactly as published:

| Dataset | Train | Validation | Test |
|---------|-------|------------|------|
| TweetEval sentiment (Twitter) | 45,615 | 2,000 | 12,284 |
| GoEmotions (Reddit) | 43,410 | 5,426 | 5,427 |

**Why:** using the official splits means (1) our numbers are directly comparable
with published papers on the same benchmarks, and (2) there is no risk of
accidentally leaking test data into training decisions. Each split has one job:

- **train** - the model learns from these examples
- **validation** - used during training to check progress after every epoch,
  to pick the best epoch (early stopping) and to tune the decision threshold.
  The model never learns from it.
- **test** - touched exactly once, at the very end, to produce the reported
  results. No decision was ever based on it.

## 2. Hyperparameters

| Setting | Value | Why |
|---------|-------|-----|
| Learning rate | **2e-5** | the standard fine-tuning value from the BERT/RoBERTa literature; small enough to adapt the pre-trained weights without destroying them |
| LR schedule | linear decay with **6% warm-up** | warm-up prevents large, destabilising updates in the first steps while the new classification head is still random; linear decay lets training settle at the end |
| Optimiser | **AdamW** | the standard optimiser for transformers - Adam with correct ("decoupled") weight decay |
| Weight decay | 0.01 | mild regularisation against overfitting |
| Effective batch size | **64 for every model** | fairness: all models see identical training dynamics (see gradient accumulation below) |
| Max sequence length | **64 tokens** | our EDA showed 95% of posts are under ~27 words, so 64 tokens keeps essentially all content at half the compute of the usual 128 |
| Max epochs | 4 (sentiment) / 6 (emotion) | the 28-label emotion task needs more passes to learn rare classes; actual epochs were chosen by early stopping |
| Early stopping patience | 2 | stop when validation stops improving for 2 evaluations - prevents overfitting and wasted compute |
| Random seed | 42 (fixed) | reproducibility: the same run always gives the same result |

Per-model batch configuration (all equal to an effective batch of 64):

| Model | Per-device batch | Gradient accumulation | Effective batch |
|-------|-----------------|----------------------|-----------------|
| DistilBERT (67M params) | 64 | 1 | 64 |
| RoBERTa (125M params) | 32 | 2 | 64 |
| DeBERTa (139M params) | 16 | 4 | 64 |

## 3. Fine-tuning techniques used, and why

### Full fine-tuning (not feature extraction)
Each model is loaded with its pre-trained weights, a new classification head
(dropout + linear layer) is added for the task, and **all** weights - the head
AND the whole encoder - are updated during training.
**Why:** full fine-tuning consistently beats freezing the encoder for models of
this size, and the models are small enough that it is affordable.

### Two different classification heads
- **Sentiment (single-label):** softmax output + cross-entropy loss - each
  tweet has exactly one correct class out of 3.
- **Emotion (multi-label):** **sigmoid** output + **binary cross-entropy**
  loss, with labels as 28-position 0/1 vectors.
  **Why:** one Reddit comment can carry several emotions at once (our EDA
  proved this), so the 28 emotions must be independent yes/no decisions,
  not one softmax competition.

### Gradient accumulation
Bigger models don't fit large batches in 16 GB of memory, so they process
smaller batches and accumulate gradients over several steps before updating
(e.g. DeBERTa: 16 x 4 = 64).
**Why:** without this, each model would train with a different effective batch
size, and the comparison between architectures would be unfair.

### Early stopping with best-model restore
After every epoch the model is evaluated on the validation set. If the score
stops improving for 2 evaluations, training stops, and the weights from the
**best** epoch (not the last one) are restored.
Selection metric: macro-F1 for sentiment; validation loss for emotion
(a fixed-threshold F1 is misleading early in multi-label training).
**Why:** transformers overfit quickly on small datasets; early stopping picks
the sweet spot automatically. In practice sentiment models used all 4 epochs,
and DeBERTa-emotion stopped at epoch 5 of 6.

### Decision-threshold tuning (multi-label only)
The emotion model outputs 28 independent probabilities. Instead of the naive
0.5 cut-off, thresholds from 0.05 to 0.50 are tried **on the validation set**
and the one with the best macro-F1 is applied once to the test set.
**Why:** with 28 mostly-rare labels, a 0.5 threshold predicts almost nothing
(near-zero recall). Threshold tuning is standard practice in the GoEmotions
literature. Crucially it is tuned on validation, never on test.

### Dynamic padding
Batches are padded to the longest text **in that batch**, not to a global
maximum (`DataCollatorWithPadding`).
**Why:** most posts are short; padding everything to 64 tokens would waste
computation on padding tokens.

### Minimal text preprocessing
Only links -> `http`, usernames -> `@user`, character floods squeezed,
whitespace normalised. Emojis and hashtags kept. No lowercasing, no stopword
removal, no stemming.
**Why:** transformers were pre-trained on raw text; aggressive classical
cleaning removes signal (emojis literally carry the emotion) and hurts accuracy.

### GPU memory cleanup between runs
After each of the 6 runs the model and trainer are deleted and the GPU cache
is emptied (`torch.mps.empty_cache()`).
**Why:** practical necessity - without it, models pile up in unified memory
and the later runs crash with out-of-memory errors on a 16 GB machine.

## 4. What was measured after fine-tuning

Every run reports, on the untouched test set: Accuracy, macro Precision,
macro Recall, **macro-F1** (headline metric - every class counts equally
despite the imbalance), plus training time and inference speed.
Each run also saves its learning curve, confusion matrix (single-label),
and the fine-tuned model itself.

## 5. Resulting epochs and scores (seed 42, full data)

| Run | Epochs trained | Macro-F1 |
|-----|----------------|----------|
| DistilBERT sentiment | 4 | 0.688 |
| RoBERTa sentiment | 4 | **0.720** |
| DeBERTa sentiment | 4 | 0.701 |
| DistilBERT emotion | 6 | 0.438 |
| RoBERTa emotion | 6 | 0.458 |
| DeBERTa emotion | 5 (early-stopped) | **0.469** |
