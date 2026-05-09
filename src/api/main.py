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
import subprocess, threading, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_pipeline():
    env = os.environ.copy()
    env['PYTHONPATH'] = '/app'
    stages = [
        ['python', 'src/ingestion/run_all.py'],
        ['python', 'src/sentiment/run.py'],
        ['python', 'src/timeseries/run.py'],
        ['python', 'src/models/run_train.py'],
        ['python', 'src/models/run_eval.py'],
    ]
    for cmd in stages:
        log.info(f'Running: {cmd}')
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/app', env=env)
        log.info(f'STDOUT: {result.stdout}')
        if result.stderr:
            log.error(f'STDERR: {result.stderr}')
        if result.returncode != 0:
            log.error(f'Stage failed: {cmd}')
            return
    log.info('Pipeline complete!')
    # Reload model after training
    global model
    if os.path.exists('models/lstm_best.pt'):
        model.load_state_dict(torch.load('models/lstm_best.pt'))
        model.eval()
        log.info('Model reloaded after training.')

@app.on_event('startup')
async def startup_event():
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
