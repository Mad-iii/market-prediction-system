# src/api/main.py
from fastapi import FastAPI
import torch, numpy as np, os
from src.models.lstm_model import LSTMModel
app = FastAPI()
model = LSTMModel(input_size=9)
if os.path.exists('models/lstm_best.pt'):
    model.load_state_dict(torch.load('models/lstm_best.pt'))
model.eval()
import pandas as pd
from fastapi.responses import HTMLResponse
import subprocess, threading

def run_pipeline():
    print('Running data pipeline...')
    subprocess.run(['python', 'src/ingestion/run_all.py'])
    subprocess.run(['python', 'src/sentiment/run.py'])
    subprocess.run(['python', 'src/timeseries/run.py'])
    print('Pipeline complete.')

@app.on_event('startup')
async def startup_event():
    if not os.path.exists('data/processed/timeseries.csv'):
        thread = threading.Thread(target=run_pipeline)
        thread.start()

@app.get('/', response_class=HTMLResponse)
def read_root():
    with open('src/api/index.html', 'r') as f:
        return f.read()

@app.get('/latest')
def get_latest():
    if os.path.exists('data/processed/timeseries.csv'):
        df = pd.read_csv('data/processed/timeseries.csv', index_col=0)
        df.index.name = 'timestamp'
        return df.tail(50).reset_index().to_dict(orient='records')
    return {'error': 'No data available - pipeline is running, check back in 2 minutes'}

@app.get('/metrics')
def get_metrics():
    import json
    if os.path.exists('data/processed/metrics.json'):
        with open('data/processed/metrics.json', 'r') as f:
            return json.load(f)
    return {'f1_score': 'N/A', 'rmse': 'N/A'}

@app.post('/predict')
def predict(data: dict):
    x = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze()).item()
    direction = 'up' if prob > 0.5 else 'down'
    return {'direction': direction, 'confidence': round(prob, 4)}
