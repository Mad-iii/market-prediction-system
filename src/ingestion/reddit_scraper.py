# src/ingestion/reddit_scraper.py
import praw
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()

def fetch_reddit_posts(subreddits=["wallstreetbets", "investing", "stocks"], limit=100):
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "market_predictor")
    )
    posts = []
    for sub in subreddits:
        for post in reddit.subreddit(sub).hot(limit=limit):
            posts.append({
                "text": post.title + " " + post.selftext,
                "created_utc": post.created_utc,
                "score": post.score
            })
    return pd.DataFrame(posts)