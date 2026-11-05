import os
import sys
from datetime import datetime

from dask import delayed, compute
from dask.distributed import Client, LocalCluster

sys.path.append("./yProv4ML")

N_PROCS = 4

@delayed
def run_job(proc, run_id):
    # simulate what SLURM / torchrun would give each process
    os.environ["SLURM_PROCID"] = str(proc)
    import torch
    torch.distributed.get_rank = lambda: proc

    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    sys.path.append("../yProv4ML")
    import yprov4ml

    PATH_DATASETS = "./data"
    BATCH_SIZE = 32
    EPOCHS = 2
    LR = 0.001
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

    yprov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name="experiment_name",
        provenance_save_dir="prov",
        save_after_n_logs=1,
        collect_all_processes=True,
        rank=proc,
        # use_run_id=run_id,
    )

    yprov4ml.log_source_code(os.path.abspath(__file__))
    yprov4ml.log_execution_command()

    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(torch.nn.Linear(28 * 28, 10))

        def forward(self, x):
            return self.model(x.view(x.size(0), -1))

    mnist_model = MNISTModel().to(DEVICE)
    yprov4ml.log_model("parent", mnist_model, is_input_to="Training_Epoch_0", is_output_to="Download")

    tform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    train_ds = Subset(MNIST(PATH_DATASETS, train=True, download=False, transform=tform), range(100))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    test_ds = Subset(MNIST(PATH_DATASETS, train=False, download=False, transform=tform), range(10))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    optim = torch.optim.Adam(mnist_model.parameters(), lr=LR)
    loss_fn = nn.MSELoss().to(DEVICE)

    yprov4ml.log_param("optimizer", "Adam", is_input_to="Training_Epoch_0")
    yprov4ml.log_param("epochs", EPOCHS, is_input_to="Training_Epoch_0")
    yprov4ml.log_param("learning_rate", LR, is_input_to="Training_Epoch_0")
    yprov4ml.log_param("batch_size", BATCH_SIZE, is_input_to="Training_Epoch_0")
    yprov4ml.log_param("loss_fn", "MSELoss", is_input_to="Training_Epoch_0")

    for epoch in range(EPOCHS):
        yprov4ml.log_dataset("validation_data", test_loader, is_input_to=f"Validation_Epoch_{epoch}", is_output_to="Download")
        yprov4ml.log_dataset("training_data", train_loader, is_input_to=f"Training_Epoch_{epoch}", is_output_to="Download")

        mnist_model.train()
        for x, y in tqdm(train_loader, desc=f"rank {proc} train {epoch}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(mnist_model(x), F.one_hot(y, 10).float())
            loss.backward()
            optim.step()

            yprov4ml.log_metric(f"MSE_T_Epoch_{epoch}", loss.item(), is_output_to=f"Training_Epoch_{epoch}", step=epoch)
            yprov4ml.log_system_metrics(is_output_to=f"Training_Epoch_{epoch}", step=epoch)

        mnist_model.eval()
        yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/checkpoint_Epoch_{epoch}", mnist_model, is_input_to=f"Validation_Epoch_{epoch}")
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"rank {proc} val {epoch}"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = loss_fn(mnist_model(x), F.one_hot(y, 10).float())
                yprov4ml.log_metric(f"MSE_V_Epoch_{epoch}", loss.item(), is_output_to=f"Validation_Epoch_{epoch}", step=epoch)

        if epoch + 1 < EPOCHS:
            yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/checkpoint_Epoch_{epoch}", mnist_model, is_input_to=f"Training_Epoch_{epoch + 1}", is_output_to=f"Training_Epoch_{epoch}")
        else:
            yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/checkpoint_Epoch_{epoch}", mnist_model, is_output_to=f"Training_Epoch_{epoch}")

    yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/weights", mnist_model, is_output_to=f"Training_Epoch_{EPOCHS - 1}")
    yprov4ml.end_run(create_graph=True, create_svg=True)


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    MNIST("./data", train=True, download=True)
    MNIST("./data", train=False, download=True)

    run_id = int(datetime.now().strftime("%Y%m%d%H%M%S"))

    cluster = LocalCluster(n_workers=N_PROCS, threads_per_worker=1, processes=True)
    client = Client(cluster)
    print(client)

    compute(*[run_job(p, run_id) for p in range(N_PROCS)])

    client.close()
    cluster.close()