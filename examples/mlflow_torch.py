import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import mlflow

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 2

mlflow.start_run()

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
mnist_model = MNISTModel()

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
mlflow.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE*4))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.0002)
mlflow.log_param("optimizer", "Adam")

for epoch in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(train_loader):
        optim.zero_grad()
        y_hat = mnist_model(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optim.step()
        mlflow.log_metric("MSE_train", loss, step=epoch*BATCH_SIZE + i)
        
for i, (x, y) in tqdm(enumerate(test_loader)):
    y_hat = mnist_model(x)
    loss = F.cross_entropy(y_hat, y)
    mlflow.log_metric("MSE_test", loss, step=epoch*BATCH_SIZE + i)

mlflow.pytorch.log_model(mnist_model, "mnist_model")
mlflow.end_run()