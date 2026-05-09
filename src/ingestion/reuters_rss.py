# src/ingestion/reuters_rss.py
import feedparser
import pandas as pd
from datetime import datetime

def fetch_news_rss():
    urls = [
        "https://news.google.com/rss/search?q=site:reuters.com+business&hl=en-US&gl=US&ceid=US:en",
        "https://search.cnbc.com/rs/search/view.xml?partnerId=2000&keywords=stock%20market",
        "http://rss.cnn.com/rss/money_latest.rss",
        "https://fortune.com/feed/"
    ]
    
    articles = []
    for url in urls:
        print(f"Fetching news from {url}...")
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                articles.append({
                    "title": getattr(entry, 'title', ''),
                    "summary": getattr(entry, 'summary', getattr(entry, 'title', '')),
                    "published": getattr(entry, 'published', datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT'))
                })
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
            
    return pd.DataFrame(articles)