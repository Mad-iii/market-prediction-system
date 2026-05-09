import os
import pandas as pd
from src.ingestion.yahoo_finance import fetch_price_data
from src.ingestion.reuters_rss import fetch_news_rss
from src.ingestion.reddit_scraper import fetch_reddit_posts
from src.ingestion.twitter_scraper import fetch_tweets

def main():
    print("Fetching price data...")
    try:
        prices = fetch_price_data()
        prices.to_csv("data/raw/prices.csv")
        print("Prices saved.")
    except Exception as e:
        print(f"Error fetching prices: {e}")

    print("Fetching News RSS feeds...")
    try:
        news = fetch_news_rss()
        news.to_csv("data/raw/reuters.csv", index=False)
        print("News saved.")
    except Exception as e:
        print(f"Error fetching news RSS: {e}")

    print("Fetching Reddit posts...")
    try:
        reddit = fetch_reddit_posts()
        reddit.to_csv("data/raw/reddit.csv", index=False)
        print("Reddit posts saved.")
    except Exception as e:
        print(f"Error fetching Reddit posts (check API keys): {e}")

    print("Fetching Tweets...")
    try:
        tweets = fetch_tweets()
        tweets.to_csv("data/raw/twitter.csv", index=False)
        print("Tweets saved.")
    except Exception as e:
        print(f"Error fetching Tweets (check API keys): {e}")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    main()
