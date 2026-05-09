import os
import pandas as pd
from src.ingestion.yahoo_finance import fetch_price_data
from src.ingestion.reuters_rss import fetch_news_rss

def main():
    print('Fetching price data...')
    try:
        prices = fetch_price_data()
        prices.to_csv('data/raw/prices.csv')
        print('Prices saved.')
    except Exception as e:
        print(f'Error fetching prices: {e}')

    print('Fetching News RSS feeds...')
    try:
        news = fetch_news_rss()
        news.to_csv('data/raw/reuters.csv', index=False)
        print('News saved.')
    except Exception as e:
        print(f'Error fetching news RSS: {e}')

if __name__ == '__main__':
    os.makedirs('data/raw', exist_ok=True)
    main()
