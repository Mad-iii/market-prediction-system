# src/sentiment/classifier.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(texts: list[str]) -> list[str]:
    results = []
    for text in texts:
        score = analyzer.polarity_scores(text)['compound']
        if score >= 0.05:
            results.append('positive')
        elif score <= -0.05:
            results.append('negative')
        else:
            results.append('neutral')
    return results
