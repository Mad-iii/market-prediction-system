# src/ingestion/reuters_rss.py
import feedparser
import pandas as pd

def fetch_reuters_news():
    url = "https://news.google.com/rss/search?q=site:reuters.com+business&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries:
        articles.append({
            "title": entry.title,
            "summary": entry.summary,
            "published": entry.published
        })
    return pd.DataFrame(articles)