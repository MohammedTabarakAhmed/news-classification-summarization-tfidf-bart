"""
Simple TF-IDF + LogisticRegression classifier helpers.
This is a minimal, runnable example. Replace dataset paths and tweak parameters as needed.
"""
import os
from typing import List
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = os.path.join("models", "classifier.pkl")


def train_classifier(csv_path: str, text_col: str = "text", label_col: str = "label") -> None:
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(csv_path)
    X = df[text_col].fillna("")
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved classifier to {MODEL_PATH}")


def load_classifier():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first using train_classifier().")
    return joblib.load(MODEL_PATH)


def predict_text(texts: List[str]):
    model = load_classifier()
    return model.predict(texts)
