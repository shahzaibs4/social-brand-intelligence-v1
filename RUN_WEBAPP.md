# How to Run the Web Application (no training needed)

This guide is for running **only the web app** - you do NOT need to download the
datasets or train anything. The app needs just three things:

1. Python with a few libraries
2. the two fine-tuned models in the `models/` folder
3. the `results/` folder (already included in the repository)

---

## 1. Install Python and the libraries

Install **Python 3.10+** (https://www.python.org/downloads/ - tick
**"Add Python to PATH"** on Windows) or Anaconda.

Then open Command Prompt (Windows) / Terminal (Mac) and install the libraries
the app needs:

```bat
pip install streamlit torch transformers pandas sentencepiece
```

(No `datasets` library needed - that is only for training.)

## 2. Get the two fine-tuned models

The app uses only **two** of the project's models:

| Folder | Model | Used for |
|--------|-------|----------|
| `models/roberta_sentiment` | fine-tuned RoBERTa (macro-F1 0.720) | sentiment |
| `models/deberta_emotion` | fine-tuned DeBERTa (macro-F1 0.469) | emotions |

They are **not** in the GitHub repository (too big, ~1 GB together), so get them
one of these ways:

**Option A - copy them (simplest).**
Copy the `models/` folder from the machine where training was done (or from the
project backup zip / a USB drive) into the project folder, so you end up with:

```
code/
├── app.py
├── results/
└── models/
    ├── roberta_sentiment/      <- config.json, model.safetensors, tokenizer files
    └── deberta_emotion/
```

**Option B - download from Hugging Face Hub (recommended).**
The models are published publicly on the Hub:

* https://huggingface.co/usman-isb/roberta-sentiment-brand
* https://huggingface.co/usman-isb/deberta-emotion-brand

Download them into the right folders with:

```bat
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('usman-isb/roberta-sentiment-brand', local_dir='models/roberta_sentiment')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('usman-isb/deberta-emotion-brand', local_dir='models/deberta_emotion')"
```

**Option C - you have no models at all.**
Then training is unavoidable: follow README.md step 5 (`notebook/models.ipynb`).
Only the `roberta_sentiment` and `deberta_emotion` runs are needed for the app.

## 3. Run the app

From the project folder:

```bat
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**
(if not, type that address into the browser yourself).

## 4. Using the app

| Page | What it does |
|------|--------------|
| **About the Project** | describes the project and its key findings |
| **Results** | all experiment tables and figures from `results/` |
| **Live Analysis** | type any post → RoBERTa predicts the sentiment, DeBERTa the emotions, with probability charts |

The **first** analysis takes ~15 seconds while the models load into memory;
every analysis after that is instant.

Stop the app with **Ctrl+C** in the terminal window.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OSError: models/roberta_sentiment ... not found` | the models are missing - do step 2 |
| Port 8501 already in use | `streamlit run app.py --server.port 8600` and open localhost:8600 |
| First analysis very slow | normal - the models are loading; wait for the spinner to finish |
| Out-of-memory on a small machine | close other apps; the app needs ~4 GB RAM |
| Results page shows no charts | the `results/` folder is missing - re-clone the repository |
