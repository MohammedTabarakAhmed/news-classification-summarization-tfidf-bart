"""
Summarization helpers using Hugging Face transformers pipeline.
This file loads a pretrained BART summarizer and exposes a simple summarize() function.
"""
from transformers import pipeline

_SUMMARIZER = None


def get_summarizer(model_name: str = "facebook/bart-large-cnn"):
    global _SUMMARIZER
    if _SUMMARIZER is None:
        _SUMMARIZER = pipeline("summarization", model=model_name)
    return _SUMMARIZER


def summarize(text: str, max_length: int = 130, min_length: int = 30):
    summarizer = get_summarizer()
    outputs = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return outputs[0]["summary_text"]
