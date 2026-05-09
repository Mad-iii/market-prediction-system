# src/sentiment/classifier.py
from transformers import pipeline

sentiment_pipeline = pipeline(
    "text-classification",
    model="ProsusAI/finbert"
)

def classify_sentiment(texts: list[str]) -> list[str]:
    results = sentiment_pipeline(texts, truncation=True, max_length=512)
    # Returns "positive", "negative", or "neutral"
    return [r["label"].lower() for r in results]