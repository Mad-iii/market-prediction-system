# 📈 Market Prediction System

An end-to-end machine learning pipeline that predicts stock market movements by combining **financial time-series data**, **news sentiment**, and **social media signals** — served via a production-ready REST API.

---

## 🧠 How It Works

The system pulls data from multiple sources, applies NLP-based sentiment analysis, trains an LSTM model on the enriched features, and exposes predictions through a FastAPI endpoint.

```
Raw Data (yfinance, RSS, Reddit, Twitter)
        ↓
  Sentiment Analysis (VADER + Transformers)
        ↓
  Time-Series Feature Engineering
        ↓
        LSTM Model Training (PyTorch)
        ↓
    FastAPI Prediction Endpoint
```

---

## 🗂️ Project Structure

```
market-prediction-system/
├── src/
│   ├── ingestion/        # Data collectors (yfinance, feeds, PRAW, Tweepy)
│   ├── sentiment/        # Sentiment scoring pipeline
│   ├── timeseries/       # Feature engineering & preprocessing
│   ├── models/           # LSTM training & evaluation
│   └── api/              # FastAPI app (main.py)
├── dags/                 # Apache Airflow DAGs for scheduling
├── data/
│   ├── raw/              # Ingested data (DVC-tracked)
│   ├── labeled/          # Sentiment-labeled data
│   └── processed/        # Model-ready features
├── models/               # Saved model checkpoints (DVC-tracked)
├── .github/workflows/    # CI/CD pipelines
├── dvc.yaml              # DVC pipeline stages
├── DOCKERFILE            # Container definition
└── requirements.txt      # Python dependencies
```

---

## ⚙️ ML Pipeline (DVC)

The pipeline is fully reproducible via [DVC](https://dvc.org/):

| Stage | Script | Description |
|---|---|---|
| `ingest` | `src/ingestion/run_all.py` | Fetch stock prices, news, Reddit & Twitter posts |
| `sentiment` | `src/sentiment/run.py` | Score each item with VADER & Transformers |
| `timeseries` | `src/timeseries/run.py` | Engineer lag features, rolling stats |
| `train` | `src/models/run_train.py` | Train LSTM model → saves `models/lstm_best.pt` |
| `evaluate` | `src/models/run_eval.py` | Run metrics on held-out test set |

Run the full pipeline with:

```bash
dvc repro
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional, recommended)

### Local Setup

```bash
git clone https://github.com/Mad-iii/market-prediction-system.git
cd market-prediction-system

pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env   # add your API keys

# Run the full DVC pipeline
dvc repro

# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t market-prediction-system .
docker run -p 8000:8000 --env-file .env market-prediction-system
```

The API will be available at `http://localhost:8000`.

---

## 🔑 Environment Variables

Create a `.env` file with the following keys:

```env
# Reddit API (PRAW)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=

# Twitter/X API (Tweepy)
TWITTER_BEARER_TOKEN=

# MLflow tracking (optional)
MLFLOW_TRACKING_URI=
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Ingestion | `yfinance`, `feedparser`, `praw`, `tweepy` |
| NLP / Sentiment | `transformers`, `vaderSentiment` |
| ML / Deep Learning | `PyTorch`, `scikit-learn`, `numpy`, `pandas` |
| Experiment Tracking | `MLflow` |
| Data Versioning | `DVC` (S3 remote) |
| Orchestration | `Apache Airflow` |
| API | `FastAPI`, `uvicorn` |
| Containerization | `Docker` |
| CI/CD | `GitHub Actions` |

---

## 📊 Model

The core model is a **Long Short-Term Memory (LSTM)** neural network trained on multi-modal features:
- Historical OHLCV price data
- Aggregated sentiment scores from news articles
- Social media signal scores (Reddit & Twitter)

Experiment tracking and model versioning are handled via **MLflow** and **DVC**.

---

## 🧪 Testing

```bash
pytest
```

---

## 📄 License

This project is open source. See `LICENSE` for details.
