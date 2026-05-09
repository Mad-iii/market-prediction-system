# src/models/train.py
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

def train_model(model, model_name, X_train, y_train, X_val, y_val,
                epochs=20, lr=0.001):

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name, "epochs": epochs, "lr": lr})

        for epoch in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb).squeeze(), yb.float())
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(X_val).squeeze()) > 0.5
        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds)

        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        mlflow.pytorch.log_model(model, model_name)
        print(f"{model_name} — Accuracy: {acc:.4f} | F1: {f1:.4f}")