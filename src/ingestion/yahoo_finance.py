import yfinance as yf

def fetch_price_data(ticker="^GSPC", period="1mo", interval="1h"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df[["Open", "High", "Low", "Close", "Volume"]]