import os
import lightning as L
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
sys.path.append("../")
import prov4ml.prov4ml as prov4ml

experiment_name = "default"
run_name = "test_runs"

prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name=experiment_name, 
    provenance_save_dir="prov", 
    mlflow_save_dir=run_name, 
)

PATH_DATASETS = "./data"
BATCH_SIZE = 32

class MNISTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        prov4ml.log_metric("MSE_train",float(loss),prov4ml.Context.TRAINING,step=self.current_epoch)
        return loss
    
    def test_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        prov4ml.log_metric("MSE_eval",loss,prov4ml.Context.VALIDATION,step=self.current_epoch)
        return loss
    
    def on_train_epoch_end(self) -> None:
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_current_execution_time("train_step", prov4ml.Context.TRAINING, self.current_epoch)

    def on_test_epoch_end(self) -> None:
        prov4ml.log_system_metrics(prov4ml.Context.VALIDATION,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.VALIDATION,step=self.current_epoch)
        prov4ml.log_current_execution_time("test_step", prov4ml.Context.VALIDATION, self.current_epoch)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)

mnist_model = MNISTModel()

# add random transformation to the dataset    
tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = torch.utils.data.Subset(train_ds, list(range(0, 500)))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

prov4ml.log_param("dataset transformation", tform)

training_params = {
    "batch_size": BATCH_SIZE,
    "lr": 0.002
}

prov4ml.log_params(training_params)

# Initialize a trainer
trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=5,
    logger=prov4ml.get_mlflow_logger(),
)

trainer.fit(mnist_model, train_loader)

# test and print the results
test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = torch.utils.data.Subset(test_ds, list(range(0, 100)))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_model(mnist_model, "mnist_model")
result = trainer.test(mnist_model, test_loader)

# get the run id for saving the provenance graph
run_id = prov4ml.get_run_id()
dot_path = f"prov/provgraph_{run_id}.dot"

prov4ml.end_run()

# run the command dot -Tsvg -O prov_graph.dot
# to generate the graph in svg format
os.system(f"dot -Tsvg -O {dot_path}")
