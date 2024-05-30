import lightning as L
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 2

# start the run in the same way as with mlflow
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov",
    mlflow_save_dir="test_runs", 
)

class MNISTModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        # prov4ml allows logging through the lightning logger --> simply add the logger and use the log method
        self.log("MSE_train", loss)
        # or with prov4ml.log_metric
        prov4ml.log_metric("MSE_train_2", loss, prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_flops_per_batch("train_flops", self, batch, prov4ml.Context.TRAINING,step=self.current_epoch)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        # change the context to EVALUATION to log the metric as evaluation metric
        prov4ml.log_metric("MSE_test",loss,prov4ml.Context.EVALUATION,step=self.current_epoch)
        return loss
    
    def on_train_epoch_end(self) -> None:
        prov4ml.log_metric("epoch", self.current_epoch, prov4ml.Context.TRAINING, step=self.current_epoch)
        # save incremental model versions
        prov4ml.save_model_version(self, f"model_version_{self.current_epoch}", prov4ml.Context.TRAINING, step=self.current_epoch)
  
        # log system and carbon metrics (once per epoch), as well as the execution time
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_current_execution_time("train_epoch_time", prov4ml.Context.TRAINING, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0002)
        prov4ml.log_param("optimizer", "Adam")
        return optim

mnist_model = MNISTModel()

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# log the dataset transformation as one-time parameter
prov4ml.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE*4))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    logger=[prov4ml.ProvMLLogger(name="mnist_model")],
    enable_checkpointing=False, 
)

trainer.fit(mnist_model, train_loader)
# log final version of the model 
# it also logs the model architecture as an artifact by default
prov4ml.log_model(mnist_model, "model_version_final")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
result = trainer.test(mnist_model, test_loader)

# save the provenance graph
prov4ml.end_run()