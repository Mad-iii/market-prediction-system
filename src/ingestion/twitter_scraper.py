# src/ingestion/twitter_scraper.py
import tweepy
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()

def fetch_tweets(query="stock market OR $SPY OR finance", max_results=100):
    client = tweepy.Client(bearer_token=os.getenv("TWITTER_BEARER_TOKEN"))
    tweets = client.search_recent_tweets(
        query=query, max_results=max_results,
        tweet_fields=["created_at", "text"]
    )
    return pd.DataFrame([{
        "text": t.text,
        "created_at": t.created_at
    } for t in tweets.data])