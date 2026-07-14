# News Classification & Summarization (TF-IDF + BART)

TL;DR
- A compact, notebook-driven project that explores news article classification using TF-IDF and performs abstractive summarization using a BART model. Everything is provided as Jupyter notebooks so you can inspect, run, and modify each step end-to-end.

Why this repo
- Demonstrates a practical pipeline for text classification (feature extraction with TF-IDF, classical ML classifiers) and neural summarization (pretrained BART from Hugging Face).
- Ideal for learning how to combine traditional NLP features with transformer-based summarizers.
- Not a packaged library — this is experimental / educational, implemented as notebooks.

Contents (high level)
- All work is in Jupyter notebooks (open the repository to view .ipynb files):
  - Notebook(s) that cover dataset loading, EDA, preprocessing, TF-IDF vectorization, classification experiments.
  - Notebook(s) that prepare inputs and run BART summarization (using Hugging Face transformers).
- Useful files:
  - Repository: https://github.com/MohammedTabarakAhmed/news-classification-summarization-tfidf-bart

Quick start — run the notebooks locally
1. Clone
   git clone https://github.com/MohammedTabarakAhmed/news-classification-summarization-tfidf-bart.git
   cd news-classification-summarization-tfidf-bart

2. Create a Python environment
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows

3. Install dependencies
   - If a requirements.txt exists:
     pip install -r requirements.txt
   - Otherwise install the common packages used in these notebooks:
     pip install jupyterlab pandas numpy scikit-learn matplotlib seaborn nltk transformers torch sentencepiece datasets rouge-score flask

4. Run Jupyter
   jupyter lab
   Open the notebooks in the repo and run cells in order (EDA → preprocessing → experiments → summarization).

Repro tip — run notebooks end-to-end from the command line
- To execute a notebook and save outputs:
  jupyter nbconvert --to notebook --execute "Notebook_Name.ipynb" --inplace

What you'll find in the notebooks
- EDA: dataset structure, class balance, example articles.
- Preprocessing: tokenization, stopwords, light cleaning, TF-IDF vectorization pipeline.
- Classification experiments: baseline models (Logistic Regression, SVM), cross-validation, metrics (precision/recall/F1), confusion matrix.
- Summarization: loading a pretrained BART model, tokenization, generating summaries (with adjustable decoding parameters), evaluation notes (ROUGE).
- Code samples to convert notebook cells to functions for reuse.

Notes & cautions
- Models and heavy transformer checkpoints should be downloaded at runtime — ensure sufficient disk space and a GPU if you plan to fine-tune.
- Do not commit large model files or dataset dumps to git. Use cloud storage (S3 or similar) if you want to persist artifacts.
- Results depend on the dataset used and hyperparameters; notebooks include places to plug your own data.

If you want this README committed to the repo
- I can create/update README.md in the repository for you if you confirm the target branch (or let it default to the repo’s default branch).

License & contact
- If you want an explicit license, add a LICENSE file (MIT is common).
- Questions or edits: MohammedTabarakAhmed (GitHub) — open an issue or PR on the repository.
