### News Classification & Summarization Pipeline

NewsLens AI is an end-to-end NLP pipeline that reads a raw news article and instantly tells you what topic it belongs to and gives you a clean short summary of it. Built with a combination of classical machine learning and modern transformer models.

---

## What it does

You give it a news article. It gives you back:
- The **category** — Politics, Sports, Technology, or Wellness
- A **confidence score** — how sure the model is
- A **summary** — 2 to 3 clean sentences describing the article

---

## How it works

The pipeline has two separate parts working together:

**Classification** takes the article text, cleans it through a preprocessing pipeline, converts it into numerical features using TF-IDF, and feeds it into a Logistic Regression model that predicts the category.

**Summarization** takes the raw article and passes it through Facebook's BART transformer model which was pretrained on CNN news articles, making it perfect for this exact use case.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Environment | Conda virtual environment |
| Dataset | HuffPost News Category Dataset (58,624 articles) |
| Preprocessing | spaCy + Gensim |
| Vectorization | TF-IDF (scikit-learn) |
| Classifier | Logistic Regression (scikit-learn) |
| Summarizer | facebook/bart-large-cnn (Hugging Face) |

---

## Preprocessing Pipeline

Every article goes through these steps before hitting the model:

1. **Whitespace normalization** — clean up messy spacing and newlines
2. **Tokenization** — split text into individual words
3. **Stop word removal** — drop words like "the", "is", "by" that carry no meaning
4. **Lemmatization** — reduce words to root form ("running" → "run", "policies" → "policy")
5. **Short token filter** — remove any token under 3 characters

---

## Model Performance

Tested on 11,725 held-out articles with an overall accuracy of **95%**.

| Category | Precision | Recall | F1 Score |
|---|---|---|---|
| POLITICS | 0.98 | 0.95 | 0.97 |
| WELLNESS | 0.93 | 0.97 | 0.95 |
| SPORTS | 0.85 | 0.91 | 0.88 |

---

## Example Output

```
Input:     "President Biden announced a $5 billion AI investment plan targeting
            universities and private companies across the United States."

Category:  POLITICS
Confidence: 0.924
Summary:   President Biden unveiled a major artificial intelligence funding
           initiative worth $5 billion. The plan targets universities and
           private sector companies to accelerate AI development nationwide.
```

---

## How to Run

```bash
# 1. Create and activate environment
conda create -p .\venv python=3.11 -y
conda activate .\venv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Open the notebook in VS Code and select the venv kernel
```

---

## Project Structure

```
NewsLens AI/
│
├── newslens.ipynb          # main notebook
├── requirements.txt        # all dependencies
├── News_Category_Dataset_v3.json  # dataset
└── README.md               # this file
```

---

*Built with Python 3.11 · spaCy · scikit-learn · Hugging Face Transformers*
