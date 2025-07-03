import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
sys.path.append("../ProvML")

import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = "mps"

# start the run in the same way as with mlflow
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
# log the dataset transformation as one-time parameter
prov4ml.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
# train_ds = Subset(train_ds, range(BATCH_SIZE*2000))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(train_loader, "train_dataset")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
# test_ds = Subset(test_ds, range(BATCH_SIZE*200))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(test_loader, "val_dataset")

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
prov4ml.log_param("optimizer", "Adam")

loss_fn = nn.MSELoss().to(DEVICE)
prov4ml.log_param("loss_fn", "MSELoss")

def tensor2str(t): 
    return str(t).replace("\n","")

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    mnist_model.train()
    mnist_model.to(DEVICE)
    for i, (x, y) in tqdm(enumerate(train_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
    
        prov4ml.log_metric("MSE_train", loss.item(), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Sample_Data", tensor2str(x), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Sample_Prediction", tensor2str(y_hat), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Sample_Label", tensor2str(y), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Sample_Gradient", tensor2str(torch.gradient(y_hat, spacing = 2.0)), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Sample_Gradient_ohe", tensor2str(torch.gradient(y, spacing = 2.0)), context=prov4ml.Context.TRAINING, step=epoch)

    mnist_model.eval()
    mnist_model.cpu()
    for i, (x, y) in tqdm(enumerate(test_loader)):
        y_hat = mnist_model(x)
        y2 = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y2)

        prov4ml.log_metric("MSE_val", loss.item(), prov4ml.Context.VALIDATION, step=epoch)
        prov4ml.log_metric("Sample_Data", tensor2str(x), context=prov4ml.Context.VALIDATION, step=epoch)
        prov4ml.log_metric("Sample_Prediction", tensor2str(y_hat), context=prov4ml.Context.VALIDATION, step=epoch)
        prov4ml.log_metric("Sample_Label", tensor2str(y2), context=prov4ml.Context.VALIDATION, step=epoch)
        prov4ml.log_metric("Sample_Gradient", tensor2str(torch.gradient(y_hat)), context=prov4ml.Context.VALIDATION, step=epoch)
        prov4ml.log_metric("Sample_Gradient_ohe", tensor2str(torch.gradient(y2)), context=prov4ml.Context.VALIDATION, step=epoch)


prov4ml.end_run(create_graph=True, create_svg=True)
