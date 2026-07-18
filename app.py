# app.py
# Brand Intelligence Web Application
#
# The practical part of the project (report Section III.F).
# It uses the two best fine-tuned models from our experiments:
#   - RoBERTa  for sentiment  (macro-F1 0.720)
#   - DeBERTa  for emotion    (macro-F1 0.469)
#
# Run it from the code/ folder with:
#   streamlit run app.py

import os
import re

import pandas as pd
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------- paths and constants ----------
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")

# use the local fine-tuned models if they exist (fast), otherwise download
# the published copies from the Hugging Face Hub (used when deployed online)
SENTIMENT_MODEL = os.path.join(MODELS, "roberta_sentiment")
if not os.path.isdir(SENTIMENT_MODEL):
    SENTIMENT_MODEL = "code-world/roberta-sentiment-brand"
EMOTION_MODEL = os.path.join(MODELS, "deberta_emotion")
if not os.path.isdir(EMOTION_MODEL):
    EMOTION_MODEL = "code-world/deberta-emotion-brand"

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
EMOTION_THRESHOLD = 0.30   # decision threshold tuned on the validation set

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu")


# ---------- helpers ----------
def clean_text(text):
    """Same minimal cleaning that was used for training."""
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource(show_spinner="Loading the fine-tuned models...")
def load_models():
    """Load both models once and keep them in memory.

    low_cpu_mem_usage avoids a temporary double copy of the weights while
    loading - important on small cloud machines (e.g. Streamlit Cloud).
    """
    sent_tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sent_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL, low_cpu_mem_usage=True).to(device).eval()
    emo_tok = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    emo_model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_MODEL, low_cpu_mem_usage=True).to(device).eval()
    return sent_tok, sent_model, emo_tok, emo_model


@st.cache_data
def load_figure_descriptions():
    """Read the figure catalog so every chart can show its explanation."""
    path = os.path.join(RESULTS, "figures_catalog.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["figure_file"], df["description"]))


def analyse(texts):
    """Run BOTH models on a list of texts.

    Returns a dataframe with the sentiment, its probabilities,
    the detected emotions and the strongest emotion per text.
    """
    sent_tok, sent_model, emo_tok, emo_model = load_models()
    cleaned = [clean_text(t) for t in texts]
    rows = []
    with torch.no_grad():
        for i in range(0, len(cleaned), 64):
            batch = cleaned[i:i + 64]

            # sentiment: softmax over 3 classes, take the highest
            enc = sent_tok(batch, truncation=True, max_length=64,
                           padding=True, return_tensors="pt").to(device)
            sent_probs = sent_model(**enc).logits.softmax(dim=-1).cpu()

            # emotion: independent sigmoid per label (multi-label)
            enc = emo_tok(batch, truncation=True, max_length=64,
                          padding=True, return_tensors="pt").to(device)
            emo_probs = emo_model(**enc).logits.sigmoid().cpu()

            for j, text in enumerate(texts[i:i + 64]):
                sp, ep = sent_probs[j], emo_probs[j]
                detected = [EMOTION_LABELS[k] for k in range(len(EMOTION_LABELS))
                            if ep[k] >= EMOTION_THRESHOLD]
                rows.append({
                    "text": text,
                    "sentiment": SENTIMENT_LABELS[int(sp.argmax())],
                    "sentiment_confidence": round(float(sp.max()), 3),
                    "emotions": ", ".join(detected) if detected else "-",
                    "top_emotion": EMOTION_LABELS[int(ep.argmax())],
                    "_sent_probs": sp.tolist(),
                    "_emo_probs": ep.tolist(),
                })
    return pd.DataFrame(rows)


def show_image(filename, title=None):
    """Show a results image with its explanation from the figure catalog."""
    path = os.path.join(RESULTS, filename)
    if os.path.exists(path):
        st.image(path, caption=title, use_column_width=True)
        desc = load_figure_descriptions().get(filename)
        if desc:
            st.caption("What this shows: " + desc)


# ---------- page setup ----------
st.set_page_config(page_title="Brand Intelligence", page_icon="📊", layout="wide")

# key="nav" stores the chosen page in the session, so the app
# always stays on the same page when it re-runs after a button click
page = st.sidebar.radio("Pages", [
    "About the Project",
    "Results",
    "Fine-Tuning Details",
    "Live Analysis",
], key="nav")
st.sidebar.markdown("---")
st.sidebar.caption(f"Sentiment model: RoBERTa (F1 0.720)\n\n"
                   f"Emotion model: DeBERTa (F1 0.469)\n\n"
                   f"Device: {device}")


# ============================================================
# Page 1 - About the Project
# ============================================================
if page == "About the Project":
    st.title("Transformer-Based Sentiment and Emotion Analysis")
    st.subheader("for Social Media Brand Intelligence")

    st.markdown("""
**The problem.** Social media produces millions of posts about brands every day.
Traditional tools only label them *positive / negative / neutral* - but knowing a
post is negative does not tell a company **why**: an angry customer needs an urgent
response, a disappointed one needs a follow-up, a fearful one needs reassurance.

**What this project does.** It builds and compares transformer models that read
social media posts and detect **both** the sentiment **and** the fine-grained
emotion behind it, then combines them into the brand-intelligence application
you are using right now.

---

#### How it was built

| Stage | Detail |
|-------|--------|
| Datasets | TweetEval (45,615 tweets, 3 sentiment classes) and GoEmotions (43,410 Reddit comments, 28 emotion labels, multi-label) - official benchmark splits |
| Models compared | DistilBERT, RoBERTa, DeBERTa - all fine-tuned with identical settings (AdamW, lr 2e-5, warm-up, early stopping, same effective batch size) |
| Evaluation | Accuracy, Precision, Recall and macro-F1 on the untouched official test sets, plus a TF-IDF + SVM baseline for context |
| Extras | Efficiency benchmark (latency / throughput), robustness analysis across the two datasets, and the emotion-beyond-sentiment analysis |

#### Key findings

1. Transformers beat the classical SVM baseline by **13-16 macro-F1 points**
2. **RoBERTa** is the best sentiment model (macro-F1 **0.720**)
3. **DeBERTa** is the best emotion model (macro-F1 **0.469**, above the published baseline)
4. A single "negative" label hides several distinct emotions - the reason this
   application shows both signals side by side
""")


# ============================================================
# Page 2 - Results
# ============================================================
elif page == "Results":
    st.title("Experimental Results")
    st.write("Every table and chart below comes directly from the experiments; "
             "an explanation is shown under each one.")

    csv = os.path.join(RESULTS, "experiments.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        # drop internal bookkeeping columns that mean nothing to the reader
        df = df.drop(columns=["quick_test"], errors="ignore")
        st.subheader("All experiment runs")
        st.dataframe(df, use_container_width=True)
        st.caption("What this shows: one row per training run on the official test sets. "
                   "'f1_macro' is the headline metric (every class counts equally, which "
                   "matters because the emotion labels are heavily imbalanced). "
                   "'threshold' is the decision cut-off tuned on the validation set for the "
                   "multi-label emotion task. 'texts_per_second' is measured inference speed.")

        st.subheader("Macro-F1 summary")
        pivot = df.pivot_table(index="model", columns="task", values="f1_macro")
        st.dataframe(pivot.style.highlight_max(axis=0, color="#c6efce"),
                     use_container_width=True)
        st.caption("What this shows: the headline comparison. Green marks the best model "
                   "per task - RoBERTa wins sentiment (0.720), DeBERTa wins emotion (0.469). "
                   "The SVM row shows how far classical machine learning falls behind: "
                   "13-16 points on both tasks.")

    col1, col2 = st.columns(2)
    with col1:
        show_image("final_comparison.png", "Models vs the classical baseline on both tasks")
        show_image("robustness_across_datasets.png", "Robustness across the two datasets")
    with col2:
        show_image("rq2_emotions_within_sentiment.png",
                   "The emotions hiding inside each sentiment class")
        eff = os.path.join(RESULTS, "efficiency.csv")
        if os.path.exists(eff):
            st.subheader("Efficiency")
            st.dataframe(pd.read_csv(eff), use_container_width=True)
            st.caption("What this shows: the speed/size side of the comparison. DistilBERT "
                       "answers in 5.2 ms and processes ~1,750 texts per second - up to 2.3x "
                       "faster than the larger models while keeping 96% of the best sentiment "
                       "quality. This is the trade-off that decides which model to deploy.")

    with st.expander("Learning curves and confusion matrices (per training run)"):
        st.caption("Learning curves: training and validation loss per epoch - both falling "
                   "and then flattening together means the model learned properly without "
                   "overfitting. Confusion matrices: where the sentiment models make mistakes - "
                   "almost all errors are between NEIGHBOURING classes (negative-neutral or "
                   "neutral-positive); direct negative-positive confusions are rare.")
        images = sorted(f for f in os.listdir(RESULTS)
                        if f.startswith(("curve_", "cm_")))
        cols = st.columns(3)
        for i, f in enumerate(images):
            with cols[i % 3]:
                st.image(os.path.join(RESULTS, f), caption=f, use_column_width=True)


# ============================================================
# Page 3 - Fine-Tuning Details
# ============================================================
elif page == "Fine-Tuning Details":
    st.title("How the Models Were Fine-Tuned")
    st.write("Each pre-trained transformer was adapted to our tasks by full fine-tuning: "
             "a new classification head is added, and ALL weights (head + encoder) are "
             "updated on the task data. Below: the exact settings and the reason for each.")

    st.subheader("Data splits")
    st.markdown("""
The **official** train / validation / test splits of each benchmark were kept unchanged:

| Dataset | Train | Validation | Test |
|---------|-------|------------|------|
| TweetEval sentiment (Twitter) | 45,615 | 2,000 | 12,284 |
| GoEmotions (Reddit) | 43,410 | 5,426 | 5,427 |

**Why:** official splits keep results comparable with published papers and make data
leakage impossible - the model learns only from *train*, the best epoch and the decision
threshold are chosen only on *validation*, and *test* is touched exactly once at the end.
""")

    st.subheader("Hyperparameters and why they were chosen")
    st.markdown("""
| Setting | Value | Why |
|---------|-------|-----|
| Optimiser | AdamW | the standard optimiser for transformers (Adam with correct weight decay) |
| Learning rate | **2e-5**, linear decay | the standard fine-tuning value - small enough to adapt the pre-trained weights without destroying them |
| Warm-up | first **6%** of steps | protects the pre-trained weights while the new random classification head settles |
| Weight decay | 0.01 | mild protection against overfitting |
| Effective batch size | **64 for every model** | fairness: all models see identical training dynamics (bigger models use gradient accumulation: 64x1, 32x2, 16x4) |
| Max sequence length | 64 tokens | our data analysis showed 95% of posts are under 27 words |
| Max epochs | 4 (sentiment) / 6 (emotion) | the 28-label emotion task needs more passes to learn rare classes |
| Early stopping | patience 2, best epoch restored | stops training when validation stops improving - prevents overfitting (DeBERTa-emotion stopped at epoch 5) |
| Random seed | 42 (fixed) | the same run always gives the same result |
""")

    st.subheader("The two classification heads")
    st.markdown("""
- **Sentiment (single-label):** softmax over 3 classes + cross-entropy loss -
  each tweet has exactly one correct class.
- **Emotion (multi-label):** **28 independent sigmoid outputs** + binary cross-entropy -
  one comment can carry several emotions at once, so each emotion is its own yes/no decision.
  The probability cut-off is **tuned on the validation set** (chosen: 0.30 for DeBERTa)
  because a naive 0.5 threshold predicts almost nothing for rare emotions.
""")

    st.subheader("Fine-tuning setup")
    fig_path = os.path.join(RESULTS, "fig_finetuning.png")
    if os.path.exists(fig_path):
        st.image(fig_path, width=430)
        st.caption("What this shows: the input post is tokenised, passes through the trainable "
                   "transformer encoder, and a new head produces the prediction. The red loop is "
                   "the AdamW weight update driven by the loss - repeated over the training data.")

    st.subheader("What fine-tuning achieved")
    st.markdown("""
| Run | Epochs trained | Macro-F1 (test) |
|-----|----------------|-----------------|
| DistilBERT sentiment | 4 | 0.688 |
| **RoBERTa sentiment** | 4 | **0.720** |
| DeBERTa sentiment | 4 | 0.701 |
| DistilBERT emotion | 6 | 0.438 |
| RoBERTa emotion | 6 | 0.458 |
| **DeBERTa emotion** | 5 (early-stopped) | **0.469** |
""")


# ============================================================
# Page 4 - Live Analysis
# ============================================================
else:
    st.title("Live Analysis")
    st.write("Type any customer post and both models will analyse it.")

    text = st.text_area("Post to analyse",
                        "The delivery was late AGAIN and support never replied. So disappointed.",
                        height=100)

    if st.button("Analyse", type="primary") and text.strip():
        result = analyse([text]).iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment")
            icon = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}[result.sentiment]
            st.metric("Prediction", f"{icon} {result.sentiment}",
                      f"confidence {result.sentiment_confidence:.0%}")
            probs = pd.Series(result._sent_probs, index=SENTIMENT_LABELS)
            st.bar_chart(probs)
            st.caption("What this chart shows: how the RoBERTa sentiment model splits its "
                       "belief between the three classes (softmax - the three bars always "
                       "sum to 1). The tallest bar is the prediction; a very tall single "
                       "bar means the model is confident.")

        with col2:
            st.subheader("Emotions")
            st.metric("Detected", result.emotions)
            probs = (pd.Series(result._emo_probs, index=EMOTION_LABELS)
                     .sort_values(ascending=False).head(8))
            st.bar_chart(probs)
            st.caption("What this chart shows: the 8 strongest of the 28 emotion "
                       "probabilities from the DeBERTa emotion model. Each emotion is an "
                       "INDEPENDENT yes/no score (sigmoid - the bars do not sum to 1, "
                       "because one post can carry several emotions). Every emotion above "
                       f"the {EMOTION_THRESHOLD:.2f} threshold - tuned on the validation "
                       "set - counts as 'detected'.")

        st.info("**Why both matter:** the sentiment tells you *how* the customer "
                "feels overall, the emotions tell you *what kind* of reaction it is - "
                "and therefore what action to take.")
