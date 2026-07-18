# Transformer-Based Sentiment and Emotion Analysis for Social Media Brand Intelligence

Compares three transformer models (**DistilBERT**, **RoBERTa**, **DeBERTa**) on two
benchmark datasets and combines the best models into a web application:

| Task | Dataset | Type |
|------|---------|------|
| Sentiment analysis | TweetEval | 3 classes (negative / neutral / positive), Twitter |
| Emotion detection | GoEmotions | 28 emotions, multi-label, Reddit |

**Final results (macro-F1, full data, official test sets):**

| Model | Sentiment | Emotion |
|-------|-----------|---------|
| SVM baseline | 0.561 | 0.345 |
| DistilBERT | 0.688 | 0.438 |
| **RoBERTa** | **0.720** | 0.458 |
| **DeBERTa** | 0.701 | **0.469** |

See `IMPLEMENTATION_FLOW.md` for the full methodology details.

> **Just want to run the web app without training?** Follow `RUN_WEBAPP.md` -
> it lists the minimal requirements (libraries + the two fine-tuned models).

---

## Project structure

```
code/
├── notebook/                 <- the pipeline, run these in order
│   ├── download_data.ipynb        1. download the datasets
│   ├── data_preprocessing.ipynb   2. clean the text
│   ├── data_eda.ipynb             3. explore the data (charts)
│   ├── models.ipynb               4. fine-tune the models (the long one)
│   └── evaluation.ipynb           5. results, baseline, analyses
├── app.py                    <- the web application
├── data/                     <- created by notebook 1 and 2
├── models/                   <- created by notebook 4
├── results/                  <- tables and figures for the report
└── IMPLEMENTATION_FLOW.md    <- detailed methodology documentation
```

---

## How to run on Windows

### 1. Install Python

Install **Anaconda** (recommended, comes with Jupyter): https://www.anaconda.com/download
— or plain **Python 3.10+** from https://www.python.org/downloads/
(tick **"Add Python to PATH"** during installation).

Check it works — open **Anaconda Prompt** (or Command Prompt):

```bat
python --version
```

### 2. Install the libraries

```bat
pip install torch transformers datasets accelerate scikit-learn pandas matplotlib sentencepiece jupyter streamlit
```

> If you have an NVIDIA graphics card, install the CUDA build of PyTorch first
> for much faster training: https://pytorch.org/get-started/locally/

### 3. Get the code

```bat
git clone https://github.com/YOUR_USERNAME/social-brand-intelligence.git
cd social-brand-intelligence
```

(or download the ZIP from GitHub and unzip it)

### 4. Download the data  (notebook 1 and 2, ~2 minutes)

Start Jupyter from the project folder:

```bat
jupyter notebook
```

Your browser opens. Then:

1. open `notebook/download_data.ipynb` → menu **Run ▸ Run All Cells**
   (downloads both datasets into `data/raw/`)
2. open `notebook/data_preprocessing.ipynb` → **Run All Cells**
   (cleans the text into `data/processed/`)
3. *(optional)* open `notebook/data_eda.ipynb` → **Run All Cells**
   (creates the data-analysis charts in `results/eda/`)

### 5. Train the models  (notebook 4 - the long step)

Open `notebook/models.ipynb`:

* **First time:** in the first code cell set `QUICK_TEST = True` and **Run All Cells**.
  This trains on tiny samples in ~15 minutes and confirms everything works on
  your machine. The scores it prints are meaningless - it is only a test.
* **Real run:** set `QUICK_TEST = False` and **Run All Cells**.
  This fine-tunes 3 models x 2 tasks on the full data.
  **It takes several hours** - run it overnight, keep the laptop
  plugged in and prevent Windows from sleeping
  (Settings ▸ System ▸ Power ▸ set "Sleep" to Never while it runs).

Notes:
* every finished run is saved immediately to `results/experiments.csv` -
  if the run is interrupted, just Run All again and it **continues where it stopped**
* the fine-tuned models are saved into `models/` (needed by the web app)

### 6. Build the results and analyses  (notebook 5, ~15 minutes)

Open `notebook/evaluation.ipynb` → **Run All Cells**.
This creates the comparison figure, summary table, SVM baseline, efficiency
table, robustness analysis and the RQ2 emotion analysis - all into `results/`.

### 7. Run the web application

From the project folder in Anaconda Prompt / Command Prompt:

```bat
streamlit run app.py
```

The browser opens at **http://localhost:8501** with three pages:

* **About the Project** - what the project is and the key findings
* **Results** - all experiment tables and figures
* **Live Analysis** - type any post and the two best models
  (RoBERTa for sentiment, DeBERTa for emotions) analyse it live

> The first analysis takes ~15 seconds while the models load; after that it is
> instant. The app needs the trained models in `models/`, so step 5 must be
> done first. Stop the app with `Ctrl+C` in the terminal.

---

## Mac / Linux

Identical steps - training automatically uses the Apple GPU (MPS) on Mac or
CUDA on Linux if available. Use Terminal instead of Command Prompt.

## Requirements

* Python 3.10 or newer
* ~16 GB RAM recommended for training (the app alone needs ~4 GB)
* ~5 GB free disk space (datasets + models)
* Internet connection for the first run (downloads datasets and pre-trained weights)
