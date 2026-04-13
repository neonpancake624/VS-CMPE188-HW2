import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

def load_data():
    X, y=load_wine(return_X_y=True)
    X =(X - X.mean(0))/X.std(0)
    return train_test_split(torch.FloatTensor(X), torch.LongTensor(y), test_size=0.2)

def build():
    return nn.Linear(13, 3)

def train(model, X, y):
    optimizer=optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    lFN = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        l= lFN(model(X), y)
        l.backward()
        optimizer.step()

def eval(model, X, y):
    with torch.no_grad():
        pred= torch.argmax(model(X), dim=1)
    return {
        "acc": accuracy_score(y, pred),
        "mse": mean_squared_error(y, pred),
        "r2": r2_score(y, pred)
    }

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()
    model= build()

    train(model, X_train, y_train)

    vm = eval(model, X_val, y_val)
    print(vm)

    success = vm["acc"] > 0.90
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