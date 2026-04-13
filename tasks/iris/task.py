import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

def get_task_metadata():
    return {"task": "iris"}

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device():
    return torch.device("cpu")

def make_dataloaders():
    X, y = load_iris(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    return (
        torch.FloatTensor(X_train), torch.LongTensor(y_train),
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )

def build():
    return nn.Linear(4, 3)

def train(model, X, y):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    lFN = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        out = model(X)
        l = lFN(out, y)
        l.backward()
        optimizer.step()

def eval(model, X, y):
    with torch.no_grad():
        out = model(X)
        pred = torch.argmax(out, dim=1)

    acc= accuracy_score(y, pred)
    mse=mean_squared_error(y, pred)
    r2=r2_score(y, pred)

    return {"acc": acc, "mse": mse, "r2": r2}

def predict(model, X):
    return torch.argmax(model(X), dim=1)

def save(*args):
    pass

if __name__ == "__main__":
    X_train, y_train, X_val, y_val = make_dataloaders()
    model= build()

    train(model, X_train, y_train)

    tm = eval(model, X_train, y_train)
    vm = eval(model, X_val, y_val)

    print(tm, vm)

    success = vm["acc"] > 0.85
    try:
        if success:
            print("pass")
            sys.exit(0)
        else:
            print("fail")
        sys.exit(1)
        
    ## used AI to help with exit code handling    
    except SystemExit as e:
        print(f"(Exit code: {e.code})")