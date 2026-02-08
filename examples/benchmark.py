

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
import time
sys.path.append("../yProv4ML")

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 3
DEVICE = "cpu"

def run(mode): 
    print(f"Running {mode}")
    start = time.time()
    if mode != "WITHOUT": 
        import yprov4ml
        yprov4ml.start_run(
            prov_user_namespace="www.example.org",
            experiment_name=f"bench_{mode}", 
            provenance_save_dir="prov",
            save_after_n_logs=100,
            collect_all_processes=False, 
            disable_codecarbon=False if "CARBON" in mode else True, 
            metrics_file_type=yprov4ml.MetricsType.CSV,
        )

    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, 10), 
                torch.nn.ReLU(),
            )

        def forward(self, x):
            return self.model(x.view(x.size(0), -1))
        
    mnist_model = MNISTModel().to(DEVICE)
    # if mode != "WITHOUT": 
    #     yprov4ml.log_model("mnist_model", mnist_model, context=yprov4ml.Context.TRAINING)

    tform = transforms.Compose([
        transforms.RandomRotation(10), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    # log the dataset transformation as one-time parameter
    if mode != "WITHOUT": 
        yprov4ml.log_param("dataset transformation", tform)

    train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
    # train_ds = Subset(train_ds, range(BATCH_SIZE*5))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    if mode != "WITHOUT": 
        yprov4ml.log_dataset("train_dataset", train_loader, context=yprov4ml.Context.TRAINING)

    optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
    if mode != "WITHOUT": 
        yprov4ml.log_param("optimizer", "Adam")

    loss_fn = nn.MSELoss().to(DEVICE)
    if mode != "WITHOUT": 
        loss_fn = yprov4ml.ProvenanceTrackedFunction(loss_fn, context=yprov4ml.Context.TRAINING)

    losses = []
    for epoch in range(EPOCHS):
        mnist_model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            y_hat = mnist_model(x)
            y = F.one_hot(y, 10).float()
            loss = loss_fn(y_hat, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        
            # log system and carbon metrics (once per epoch), as well as the execution time
            if "METRIC" in mode: 
                yprov4ml.log_metric("MSE", loss.item(), context=yprov4ml.Context.TRAINING, step=epoch)
            if "SYSTEM" in mode: 
                yprov4ml.log_system_metrics(yprov4ml.Context.TRAINING, step=epoch)
            if "CARBON" in mode: 
                yprov4ml.log_carbon_metrics(yprov4ml.Context.TRAINING, step=epoch)
            # yprov4ml.log_flops_per_batch("test", mnist_model, (x, y), yprov4ml.Context.TRAINING, step=epoch)
        # save incremental model versions
        if mode != "WITHOUT": 
            yprov4ml.save_model_version(f"mnist_model_version", mnist_model, yprov4ml.Context.TRAINING, epoch)
    if mode != "WITHOUT": 
        yprov4ml.end_run(create_graph=False, create_svg=False, crate_ro_crate=False)

    end = time.time()
    print(f"Time for {mode}: {end-start}")

def main(): 
    MODES = ["WITHOUT", "PROV_ONLY", "PROV+1METRIC", "PROV+SYSTEM", "PROV+CARBON", "PROV+SYSTEM+CARBON"]
    for mode in MODES: 
        run(mode)

if __name__ == "__main__": 
    main()