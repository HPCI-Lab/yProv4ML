import lightning as L
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

import mlflow
from pytorch_lightning.loggers import MLFlowLogger

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 2

mlflow.start_run()

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
        self.log("MSE_train", loss)
        mlflow.log_metric("MSE_train_2", loss, step=self.current_epoch)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        mlflow.log_metric("MSE_test",loss,step=self.current_epoch)
        return loss
    
    def on_train_epoch_end(self) -> None:
        mlflow.log_metric("epoch", self.current_epoch, step=self.current_epoch)
        # all system and ee metrics would require a custom implementation
  
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0002)
        mlflow.log_param("optimizer", "Adam")
        return optim

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

trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    logger=[MLFlowLogger()], 
    enable_checkpointing=False, 
)

trainer.fit(mnist_model, train_loader)
mlflow.pytorch.log_model(mnist_model, "mnist_model")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
result = trainer.test(mnist_model, test_loader)

mlflow.end_run()