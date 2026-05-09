import os
import pandas as pd
import torch
from src.models.lstm_model import LSTMModel
from src.evaluations.metrices import evaluate_classification

def main():
    if not os.path.exists("data/processed/timeseries.csv"):
        print("Data not found.")
        return
    
    df = pd.read_csv("data/processed/timeseries.csv", index_col=0)
    # Simple evaluation on the last few rows
    features = ["Open", "High", "Low", "Close", "Volume", "positive", "negative", "neutral", "sentiment_score"]
    data = torch.tensor(df[features].values, dtype=torch.float32).unsqueeze(1) # [seq, batch, feat]
    
    # Load model
    model = LSTMModel(input_size=9)
    if os.path.exists("models/lstm_best.pt"):
        model.load_state_dict(torch.load("models/lstm_best.pt"))
    model.eval()
    
    with torch.no_grad():
        # This is just a dummy eval for the DAG flow
        outputs = model(data)
        preds = (torch.sigmoid(outputs) > 0.5).squeeze().numpy()
        y_true = df["direction"].values
        
        # Ensure same length for metrics
        min_len = min(len(preds), len(y_true))
        metrics = evaluate_classification(y_true[:min_len], preds[:min_len])
        print(f"Evaluation Metrics: {metrics}")

if __name__ == "__main__":
    main()
