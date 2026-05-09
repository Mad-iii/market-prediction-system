import os
import pandas as pd
from src.timeseries.builder import build_timeseries

def main():
    if not os.path.exists("data/raw/prices.csv"):
        print("Price data not found. Run ingestion first.")
        return
    
    price_df = pd.read_csv("data/raw/prices.csv", index_col=0)
    
    # Load all labeled sentiment data
    sentiment_files = [
        "data/labeled/reuters_labeled.csv",
        "data/labeled/reddit_labeled.csv",
        "data/labeled/twitter_labeled.csv"
    ]
    
    dfs = []
    for f in sentiment_files:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f))
    
    if not dfs:
        print("No sentiment data found. Run sentiment labeling first.")
        return
    
    sentiment_df = pd.concat(dfs)
    
    print("Building time-series dataset...")
    ts_df = build_timeseries(price_df, sentiment_df)
    
    os.makedirs("data/processed", exist_ok=True)
    ts_df.to_csv("data/processed/timeseries.csv")
    print("Time-series dataset saved to data/processed/timeseries.csv")

if __name__ == "__main__":
    main()
