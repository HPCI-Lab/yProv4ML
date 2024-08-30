import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 1
DEVICE = "mps"

# start the run in the same way as with mlflow
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
)

# prov4ml.register_final_metric("MSE_test", 10, prov4ml.FoldOperation.MIN)
# prov4ml.register_final_metric("MSE_train", 10, prov4ml.FoldOperation.MIN)
# prov4ml.register_final_metric("emissions_rate", 0.0, prov4ml.FoldOperation.ADD)

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
# train_ds = Subset(train_ds, range(BATCH_SIZE*4))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(train_loader, "train_dataset")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
# test_ds = Subset(test_ds, range(BATCH_SIZE*2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(test_loader, "train_dataset")

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
prov4ml.log_param("optimizer", "Adam")

loss_fn = nn.MSELoss().to(DEVICE)

losses = []
for epoch in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        prov4ml.log_metric("MSE_train", loss, context=prov4ml.Context.TRAINING, step=epoch)
        losses.append(loss.item())
    
    # log system and carbon metrics (once per epoch), as well as the execution time
    prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch)
    prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)
    # save incremental model versions
    prov4ml.save_model_version(mnist_model, f"mnist_model_version_{epoch}", prov4ml.Context.TRAINING, epoch)

import numpy as np   
cm = np.zeros((10, 10))
acc = 0

mnist_model.eval()
mnist_model.cpu()
for i, (x, y) in tqdm(enumerate(test_loader)):
    y_hat = mnist_model(x)
    y2 = F.one_hot(y, 10).float()
    loss = loss_fn(y_hat, y2)

    # add confusion matrix
    y_pred = torch.argmax(y_hat, dim=1)
    for j in range(y.shape[0]):
        cm[y[j], y_pred[j]] += 1
    # change the context to EVALUATION to log the metric as evaluation metric
    prov4ml.log_metric("MSE_test", loss, prov4ml.Context.EVALUATION, step=epoch)

# log final version of the model 
# it also logs the model architecture as an artifact by default
prov4ml.log_model(mnist_model, "mnist_model_final")

# save the provenance graph
prov4ml.end_run(create_graph=True, create_svg=True)

# log the losses as a graph
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.lineplot(x=range(len(losses)), y=losses)
plt.savefig("losses.png")

# plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="g")
plt.savefig("confusion_matrix.png")
