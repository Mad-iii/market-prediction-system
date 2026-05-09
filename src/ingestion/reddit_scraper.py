# src/ingestion/reddit_scraper.py
import praw
import pandas as pd

def fetch_reddit_posts(subreddits=["wallstreetbets", "investing", "stocks"], limit=100):
    reddit = praw.Reddit(
        client_id="YOUR_ID",
        client_secret="YOUR_SECRET",
        user_agent="market_predictor"
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