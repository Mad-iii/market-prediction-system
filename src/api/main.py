# src/api/main.py
from fastapi import FastAPI
import torch, numpy as np
from src.models.lstm_model import LSTMModel

app = FastAPI()
model = LSTMModel(input_size=8)
model.load_state_dict(torch.load("models/lstm_best.pt"))
model.eval()

@app.post("/predict")
def predict(data: dict):
    # data = {"features": [[...], [...], ...]}  # sequence of hourly features
    x = torch.tensor(data["features"], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x).squeeze()).item()
    direction = "up" if prob > 0.5 else "down"
    return {"direction": direction, "confidence": round(prob, 4)}