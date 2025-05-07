import lightning as L
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append("../yProvML")

import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 2

class MNISTModel(LightningModule):
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
        self.log("MSE_train", loss.item(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        # return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("MSE_val", loss)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("MSE_test",loss)
        return loss
    
    def on_train_epoch_end(self) -> None:
        # prov4ml.log_metric("epoch", self.current_epoch, prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.save_model_version(self, f"model_version_{self.current_epoch}", prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_current_execution_time("train_epoch_time", prov4ml.Context.TRAINING, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0002)
        prov4ml.log_param("optimizer", optim)
        return optim


mnist_model = MNISTModel()

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# log the dataset transformation as one-time parameter
prov4ml.log_param("dataset_transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
val_ds = Subset(train_ds, range(BATCH_SIZE * 1))
train_ds = Subset(train_ds, range(BATCH_SIZE * 10))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

prov4ml.log_dataset(train_loader, "train_dataset")
prov4ml.log_dataset(val_loader, "val_dataset")

trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCHS,
    logger=[prov4ml.ProvMLLogger()],
    enable_checkpointing=False, 
    log_every_n_steps=1
)

trainer.fit(mnist_model, train_loader, val_dataloaders=val_loader)
prov4ml.log_model(mnist_model, "model_version_final")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE * 2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

prov4ml.log_dataset(test_loader, "test_dataset")

result = trainer.test(mnist_model, test_loader)