import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

torch.manual_seed(42)
np.random.seed(42)

def make_dataloaders():
    X, y = fetch_california_housing(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    y = y.reshape(-1,1)
    return train_test_split(torch.FloatTensor(X), torch.FloatTensor(y), test_size=0.2)

def build_model():
    return nn.Linear(8, 1)

def train(model, X, y):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(200):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

def evaluate(model, X, y):
    with torch.no_grad():
        preds = model(X)
    return {
        "mse": mean_squared_error(y, preds),
        "r2": r2_score(y, preds),
        "mae": mean_absolute_error(y, preds)
    }

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = make_dataloaders()

    model = build_model()
    train(model, X_train, y_train)

    val_metrics = evaluate(model, X_val, y_val)

    # sklearn baseline
    lr = LinearRegression().fit(X_train, y_train)
    sk_preds = lr.predict(X_val)
    sk_r2 = r2_score(y_val, sk_preds)

    print("Torch:", val_metrics)
    print("Sklearn R2:", sk_r2)

    success = val_metrics["r2"] > 0.5
    sys.exit(0 if success else 1)