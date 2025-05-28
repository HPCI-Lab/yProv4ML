import time
start_time = time.time()
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
sys.path.append("../yProvML")
import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = "cpu"

prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
)

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
prov4ml.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(train_loader, "train_dataset")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(test_loader, "val_dataset")

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss().to(DEVICE)
prov4ml.log_param("optimizer", "Adam")
prov4ml.log_param("loss_fn", "MSELoss")

losses = []
for epoch in range(EPOCHS):
    mnist_model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    
        prov4ml.log_metric("Loss", loss.item(), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)
        print(psutil.virtual_memory().percent)
    prov4ml.save_model_version(mnist_model, "mnist_model_version",prov4ml.Context.TRAINING)
    
    mnist_model.eval()
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = mnist_model(x)
        y2 = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y2)

        prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.VALIDATION, step=epoch)
        print(psutil.virtual_memory().percent)

prov4ml.log_model(mnist_model, "mnist_model_final")
prov4ml.end_run(create_graph=True, create_svg=True)

endtime = time.time()
print(f"Total time: {endtime - start_time}")