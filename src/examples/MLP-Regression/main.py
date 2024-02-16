import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

import mlflow
import prov4ml.prov4ml as prov4ml
"""
https://machinelearningmastery.com/building-a-regression-model-in-pytorch/
"""
mlflow.set_experiment("MLP-Regression")
with prov4ml.start_run(prov_user_namespace="www.example.org",run_name="run"):
    # Read data
    data = fetch_california_housing()
    X, y = data.data, data.target

    # train-test split for model evaluation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

    mlflow.log_input(mlflow.data.numpy_dataset.from_numpy(X_train_raw),context="training")
    mlflow.log_input(mlflow.data.numpy_dataset.from_numpy(X_test_raw), context="testing")

    # Standardizing data
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Define the model
    model = nn.Sequential(
        nn.Linear(8, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 5   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
    mlflow.log_params({
        "n_epochs":n_epochs,
        "batch_size":batch_size,
        "loss_fn":loss_fn._get_name(),
        "optimizer":optimizer.__class__.__name__,
        "lr":0.0001
    })
    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        print(f"Epoch {epoch}: MSE = {mse:.2f}")

        prov4ml.log_metric("MSE",mse,prov4ml.Context.TRAINING,step=epoch)
        
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))

    mlflow.pytorch.log_model(pytorch_model=model,artifact_path="mlp",registered_model_name="mlp")
    model.eval()
    with torch.no_grad():
        # Test out inference with 5 samples
        for i in range(5):
            X_sample = X_test_raw[i: i+1]
            X_sample = scaler.transform(X_sample)
            X_sample = torch.tensor(X_sample, dtype=torch.float32)
            y_pred = model(X_sample)
            print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")