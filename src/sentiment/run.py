import os
import pandas as pd
from src.sentiment.classifier import classify_sentiment

def process_sentiment(input_file, output_file, text_col):
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Skipping.")
        return

    df = pd.read_csv(input_file)
    if df.empty:
        print(f"File {input_file} is empty. Skipping.")
        return

    print(f"Classifying sentiment for {input_file}...")
    texts = df[text_col].astype(str).tolist()
    df["sentiment"] = classify_sentiment(texts)
    
    # Standardize column names for time-series builder
    if "published" in df.columns:
        df["timestamp"] = df["published"]
    elif "created_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["created_utc"], unit="s")
    elif "created_at" in df.columns:
        df["timestamp"] = df["created_at"]
    
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

def main():
    os.makedirs("data/labeled", exist_ok=True)
    
    # Process news
    process_sentiment("data/raw/reuters.csv", "data/labeled/reuters_labeled.csv", "summary")
    
    # Process reddit
    process_sentiment("data/raw/reddit.csv", "data/labeled/reddit_labeled.csv", "text")
    
    # Process twitter
    process_sentiment("data/raw/twitter.csv", "data/labeled/twitter_labeled.csv", "text")

if __name__ == "__main__":
    main()
