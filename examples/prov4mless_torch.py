# import time
# start_time = time.time()
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = "cpu"

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
mnist_model = MNISTModel().to(DEVICE)

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
# train_ds = Subset(train_ds, range(1000 * BATCH_SIZE))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
# test_ds = Subset(test_ds, range(100 * BATCH_SIZE))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)

loss_fn = nn.MSELoss().to(DEVICE)

losses = []
for epoch in range(EPOCHS):
    mnist_model.train()
    for i, (x, y) in enumerate(train_loader):
    # for i, (x, y) in tqdm(enumerate(train_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        print(psutil.virtual_memory().percent)
    
    torch.save(mnist_model, f"test_{epoch}.pth")

    mnist_model.eval()
    for i, (x, y) in enumerate(test_loader):
    # for i, (x, y) in tqdm(enumerate(test_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = mnist_model(x)
        y2 = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y2)
        print(psutil.virtual_memory().percent)

torch.save(mnist_model, "test_final.pth")

# endtime = time.time()

# print(f"Total time: {endtime - start_time}")