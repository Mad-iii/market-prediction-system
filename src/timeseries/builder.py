# src/timeseries/builder.py
import pandas as pd

def build_timeseries(price_df, sentiment_df):
    # sentiment_df must have: timestamp, sentiment (pos/neg/neu)
    sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"])
    sentiment_df = sentiment_df.set_index("timestamp").resample("1H").agg(
        positive=("sentiment", lambda x: (x == "positive").sum()),
        negative=("sentiment", lambda x: (x == "negative").sum()),
        neutral=("sentiment",  lambda x: (x == "neutral").sum()),
    )
    sentiment_df["sentiment_score"] = (
        sentiment_df["positive"] - sentiment_df["negative"]
    ) / (sentiment_df["positive"] + sentiment_df["negative"] + sentiment_df["neutral"] + 1e-9)

    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    price_df = price_df.resample("1H").last()

    merged = price_df.join(sentiment_df, how="inner")
    merged.dropna(inplace=True)

    # Target: market direction (1 = up, 0 = down)
    merged["direction"] = (merged["Close"].shift(-1) > merged["Close"]).astype(int)
    return merged